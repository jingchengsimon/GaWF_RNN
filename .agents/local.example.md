# Local Agent Configuration

Copy this file to `.agents/local.md` and fill in machine-specific values. The local file is
ignored by Git. Do not put passwords, tokens, or private keys here.

## SSH aliases

- Amarel: `<ssh-alias>`
- Legacy Amarel, if needed: `<ssh-alias>`
- sjc remote: `<ssh-alias>`

## Remote project roots

- Amarel project: `<absolute-path>`
- Optional alternate Amarel worktree: `<absolute-path>`
- sjc project: `<absolute-path>`

## Data roots

- Amarel stimuli/data: `<absolute-path>`
- sjc stimuli/data: `<absolute-path>`

## Conda

- Environment: `aim3_rnn`
- Amarel initialization script: `<absolute-path-to-conda.sh>`
- sjc initialization script: `<absolute-path-to-conda.sh>`

## Wrapper configuration

- sjc wrapper config: `remote/config.sh`
