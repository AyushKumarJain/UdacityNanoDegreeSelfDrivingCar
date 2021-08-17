#!/usr/bin/env python
"""Create AMPL demo packages.

Usage:
  create-demo-packages.py [--cache] [update]

Options:
  --cache  Cache downloaded packages (for debugging).
"""

from __future__ import print_function
import fileutil, gzip, os, shutil, stat, subprocess
import tarfile, tempfile, timer, urllib, zipfile
from docopt import docopt
from glob import glob
from sets import Set
from StringIO import StringIO
from urlparse import urlparse

# URL for downloading student versions of AMPL binaries.
student_url = 'http://www.ampl.com/netlib/ampl/student/'

# URL for downloading a command-line version of AMPL.
amplcml_url = 'http://www.ampl.com/NEW/TABLES/amplcml.zip'

# URL for downloading AMPL table handler.
ampltabl_url = 'http://www.ampl.com/NEW/TABLEPROXY/'

# Map from system names used for demo packages to ampltabl's system names.
ampltabl_sys = {
  'linux32': 'linux-intel32',
  'linux64': 'linux-intel64',
  'macosx':  'macosx64',
  'mswin':   'mswin32'
}

# Files to download.
download_files = [
  'README', 'ampl.gz', 'cplex.gz', 'gjh.gz',
  'gurobi.tgz', 'minos.gz', 'snopt.gz']

# Files that need executable permission.
executables = Set(['ampl', 'cplex', 'gjh', 'gurobix', 'minos', 'snopt'])

# Paths to files and directories to copy from amplcml.zip to the demo package.
extra_paths = ['MODELS/', 'kestrel', 'modinc']

# Writes a file object f to the file with specified name.
def writefile(f, filename):
  with open(filename, 'wb') as out:
    out.write(f.read())

cache_dir = 'cache'

# Retrieve the url or use cached version of the file if available.
def retrieve_cached(url, system = None):
  filename = os.path.basename(urlparse(url).path)
  cached_path = cache_dir
  if system is not None:
    cached_path = os.path.join(cache_dir, system)
    if not os.path.exists(cached_path):
      os.mkdir(cached_path)
  cached_path = os.path.join(cached_path, filename)
  if os.path.exists(cached_path):
    print('Using cached version of', filename)
  else:
    print('Downloading', filename)
    urllib.urlretrieve(url, cached_path)
  return cached_path

# Extract files from amplcml.zip.
def extract_amplcml(amplcml, ampl_demo_dir, extra_paths = None):
  for name in amplcml.namelist():
    if extra_paths is not None:
      found = False
      for path in extra_paths:
        if name.startswith('amplcml/' + path):
          found = True
          break
      if not found:
        continue
    outname = name.replace('amplcml/', ampl_demo_dir + '/')
    if name.endswith('/'):
      os.makedirs(outname)
    else:
      writefile(amplcml.open(name), outname)

# Prepare a demo package for a UNIX-like system.
def prepare_unix_package(amplcml, ampl_demo_dir, kestrel, system):
  os.mkdir(ampl_demo_dir)
  kestrel.extractall(ampl_demo_dir)
  os.rename(os.path.join(ampl_demo_dir, 'README'),
            os.path.join(ampl_demo_dir, 'README.kestrel'))
  extract_amplcml(amplcml, ampl_demo_dir, extra_paths)

  # Download ampl and solvers.
  for filename in download_files:
    sysdir = system
    if system == 'macosx':
      sysdir = system + '/x86_32'
    elif system == 'linux32':
      sysdir = 'linux'
    retrieved_file = retrieve_cached(
      '{}/{}/{}'.format(student_url, sysdir, filename), system)
    # Unpack if necessary.
    outfilename = filename
    if filename.endswith('.gz'):
      outfilename = filename.replace('.gz', '')
      with gzip.GzipFile(retrieved_file) as f:
        writefile(f, os.path.join(ampl_demo_dir, outfilename))
    elif filename.endswith('.tgz'):
      with tarfile.open(retrieved_file) as tar:
        tar.extractall(ampl_demo_dir)
    else:
      shutil.copy(retrieved_file, os.path.join(ampl_demo_dir, filename))
    # Add executable permissions.
    if outfilename in executables:
      path = os.path.join(ampl_demo_dir, outfilename)
      st = os.stat(path)
      os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

  # Replace libgurobi*.so link with the library ligurobi.so* because some
  # programs don't support symlinks in zip archives.
  libs = glob(os.path.join(ampl_demo_dir, 'libgurobi.so*'))
  if len(libs) > 0:
    libgurobi = libs[0]
    libgurobi_link = glob(os.path.join(ampl_demo_dir, 'libgurobi*.so'))[0]
    os.remove(libgurobi_link)
    shutil.move(libgurobi, libgurobi_link)

  # Download ampltabl.dll.
  url = ampltabl_url + 'ampltabl.{}.tgz'.format(ampltabl_sys[system])
  with gzip.GzipFile(retrieve_cached(url)) as f:
    writefile(f, os.path.join(ampl_demo_dir, 'ampltabl.dll'))

def package(basename, archive_format, package_dir):
  with timer.print_time('Creating', basename, 'package'):
    if archive_format == 'gztar':
      # Use command-line tar instead of shutil.make_archive because the
      # latter is too slow.
      archive = os.path.join(os.getcwd(), basename + '.tar.gz')
      command = ['tar', 'czf', archive] + os.listdir(package_dir)
      subprocess.check_call(command, cwd=package_dir)
    else:
      shutil.make_archive(basename, archive_format, package_dir, '.')

# Map from system name to IDE package suffix.
sys2ide = {
  'linux32': 'linux32.tgz',
  'linux64': 'linux64.tgz',
  'macosx':  'mac64.tgz',
  'mswin':   'win32.zip'
}

if __name__ == '__main__':
  args = docopt(__doc__)
  
  # Create packages.
  workdir = tempfile.mkdtemp()
  try:
    if not args['--cache']:
      cache_dir = os.path.join(workdir, 'cache')
    if not os.path.exists(cache_dir):
      os.mkdir(cache_dir)
    package_dir = os.path.join(workdir, 'package')
    os.mkdir(package_dir)
    amplcml = zipfile.ZipFile(retrieve_cached(amplcml_url))
    ampl_demo_dir = os.path.join(package_dir, 'ampl-demo')
    # Download kestrel.
    kestrel_tar = tarfile.open(
      retrieve_cached('http://ampl.com/dl/neos/kestrel.tar.gz'), 'r:gz')
    for system in ['linux32', 'linux64', 'macosx', 'mswin']:
      # Prepare the command-line demo package.
      if system != 'mswin':
        archive_format = 'gztar'
        prepare_unix_package(amplcml, ampl_demo_dir, kestrel_tar, system)
      else:
        archive_format = 'zip'
        extract_amplcml(amplcml, ampl_demo_dir)
        url = 'http://ampl.com/dl/neos/kestrel.zip'
        with zipfile.ZipFile(retrieve_cached(url)) as f:
          f.extractall(ampl_demo_dir)
      basename = 'ampl-demo-' + system
      package(basename, archive_format, package_dir)

      # Prepare the IDE demo package.
      amplide_demo_dir = os.path.join(package_dir, 'amplide-demo')
      amplide_url = 'http://www.ampl.com/dl/IDE/amplide.' + sys2ide[system]
      amplide = retrieve_cached(amplide_url)
      iszip = amplide_url.endswith('zip')
      archive_open = zipfile.ZipFile if iszip else tarfile.open
      with archive_open(amplide) as archive:
        archive.extractall(package_dir)
      shutil.move(os.path.join(package_dir, 'amplide'), amplide_demo_dir)
      shutil.move(ampl_demo_dir, os.path.join(amplide_demo_dir, 'ampl'))
      basename = 'amplide-demo-' + system
      package(basename, archive_format, package_dir)
      shutil.rmtree(amplide_demo_dir)
  finally:
    shutil.rmtree(workdir)

  # Move packages to the server.
  if args['update']:
    demo_dir = '/var/www/dl/demo'
    for system in ['linux32', 'linux64', 'macosx', 'mswin']:
      ext = '.zip' if system == 'mswin' else '.tar.gz'
      fileutil.move('ampl-demo-' + system + ext, demo_dir)
      fileutil.move('amplide-demo-' + system + ext, demo_dir)
