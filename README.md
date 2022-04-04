## Introduction

EMA Module is a module for context-aware EMA triggering.

## Usage:

```Python
from pathlib import Path
import EMAModule

mod = EMAModule.MainModule()
datapath = '/PathTo/data_uniterctXXX-2021-12-15-18-26-45.csv'
user_id = 'uniterctXXX'

# Here is where Sample_{username}.csv, density_{username}.npy, bndrs_{username}.csv and userinfo_{username}.json files are stored.
filepath = Path('/PathTo/User/Metadata/Files/')

mod.main(datapath, filepath, user_id)
```

## Local Installation
```bash
python3 setup.py install
```

## Deploying to UNITE server:

Remember to change env.hosts in fabfile.py for deployment to server before running commands.

Use Python fabricator:
```bash
fab deploy
```
