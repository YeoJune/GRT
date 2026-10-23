"""Verified file transfers and the optional shared prepared-data cache."""
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from urllib.parse import urlsplit
import zipfile

from grt.utils.server import ServerError, digest


def transfer_url(url):
    parsed = urlsplit(url)
    if parsed.username or parsed.password or not (parsed.scheme == 'https' or
        parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')):
        raise ValueError('Insecure server file URL')
    return url


def download_artifact(client, artifact, destination):
    import requests
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    link = client.request('GET', '/artifacts/' + artifact['id'] + '/download')
    fd, name = tempfile.mkstemp(dir=destination.parent, prefix='.download-')
    temporary = Path(name)
    value, size = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, 'wb') as stream, requests.get(transfer_url(link['url']),
                stream=True, timeout=(5, 120), allow_redirects=False) as response:
            if response.status_code != 200:
                raise ServerError(response.status_code, 'download_failed')
            for chunk in response.iter_content(1024 * 1024):
                size += len(chunk)
                if size > artifact['size_bytes']:
                    raise ValueError('Download exceeds declared size')
                value.update(chunk)
                stream.write(chunk)
        if size != artifact['size_bytes'] or value.hexdigest() != artifact['sha256']:
            raise ValueError('Downloaded artifact size/SHA256 mismatch')
        temporary.replace(destination)
    except requests.RequestException:
        raise ServerError(0, 'download_unavailable') from None
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepared_files(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    files = {split + '.pt': sha for split, sha in manifest['sha256'].items()}
    files.update(manifest.get('prepared_artifacts_sha256', {}))
    if 'vocab_sha256' in manifest:
        files['vocab.json'] = manifest['vocab_sha256']
    for name, sha in files.items():
        parts = PurePosixPath(name)
        path = directory / name
        if parts.is_absolute() or '..' in parts.parts or '\\' in name or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Invalid prepared file path')
        if digest(path) != sha:
            raise ValueError('Prepared file SHA256 mismatch: ' + name)
    return {'manifest.json', *files}


def cache_prepared(client, directory):
    import requests
    directory = Path(directory)
    files = prepared_files(directory)
    sha = digest(directory / 'manifest.json')
    try:
        return client.request('GET', '/datasets/' + sha)
    except ServerError as error:
        if error.status != 404:
            raise
    with tempfile.TemporaryDirectory(prefix='grt-prepared-') as temporary:
        archive = Path(temporary) / ('prepared-' + sha + '.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
            for name in sorted(files):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                with bundle.open(info, 'w', force_zip64=True) as out, (directory / name).open('rb') as source:
                    shutil.copyfileobj(source, out, 1024 * 1024)
        item = client.request('POST', '/artifacts/uploads', dict(name=archive.name, kind='dataset',
            size_bytes=archive.stat().st_size, sha256=digest(archive), format='zip', format_version='1',
            extensions=dict(cache_format='grt.prepared/1', manifest_sha256=sha)))
        if item['status'] == 'reserved':
            try:
                link = client.request('POST', '/artifacts/uploads', {'upload_id': item['id']})['upload']
            except ServerError as error:
                if error.code != 'upload_closed':
                    raise
                link = None  # A concurrent publisher may already have completed this archive.
            if link:
                try:
                    with archive.open('rb') as stream:
                        response = requests.put(transfer_url(link['url']), data=stream, headers=link['headers'],
                            timeout=(5, 120), allow_redirects=False)
                except requests.RequestException:
                    raise ServerError(0, 'upload_unavailable') from None
                if response.status_code not in (200, 201, 204, 409):
                    raise ServerError(response.status_code, 'upload_failed')
        client.request('POST', '/artifacts/uploads/' + item['id'] + '/complete', {})
        until = time.monotonic() + 600
        while True:
            item = client.request('GET', '/artifacts/uploads/' + item['id'])
            if item['status'] == 'verified':
                return item
            if item['status'] == 'failed' or time.monotonic() > until:
                raise ServerError(422, 'cache_not_verified')
            time.sleep(.5)


def fetch_prepared(client, sha, destination):
    if not isinstance(sha, str) or not re.fullmatch('[a-f0-9]{64}', sha):
        raise ValueError('Prepared cache requires a manifest SHA256')
    destination = Path(destination)
    if destination.exists():
        if digest(destination / 'manifest.json') != sha:
            raise ValueError('Existing prepared manifest differs from requested cache')
        prepared_files(destination)
        return destination
    item = client.request('GET', '/datasets/' + sha)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.prepared-') as temporary:
        root = Path(temporary)
        download_artifact(client, item, root / 'prepared.zip')
        unpacked = root / 'data'
        unpacked.mkdir()
        with zipfile.ZipFile(root / 'prepared.zip') as bundle:
            names = set()
            for member in bundle.infolist():
                name = PurePosixPath(member.filename)
                if (name.is_absolute() or '..' in name.parts or '\\' in member.filename or
                    member.is_dir() or member.compress_type != zipfile.ZIP_STORED or member.filename in names or
                    (member.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError('Unsafe prepared archive entry')
                names.add(member.filename)
            if sum(m.file_size for m in bundle.infolist()) > item['size_bytes']:
                raise ValueError('Prepared archive must be uncompressed')
            bundle.extractall(unpacked)
        if digest(unpacked / 'manifest.json') != sha:
            raise ValueError('Cached manifest SHA256 mismatch')
        if prepared_files(unpacked) != names:
            raise ValueError('Prepared archive contains untracked files')
        unpacked.rename(destination)
    return destination
