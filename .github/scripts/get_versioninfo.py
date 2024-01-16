#!/usr/bin/env python3

import os, sys
from build.util import project_wheel_metadata

if 'GITHUB_OUTPUT' in os.environ:
    file = open(os.environ['GITHUB_OUTPUT'], 'w')
else:
    file = sys.stdout

msg = project_wheel_metadata('.')
version = msg.get('version')
project = msg.get('name')

print(f'project={project}', file=file)
print(f'version={version}', file=file)
print(f'tag=v{version}', file=file)
