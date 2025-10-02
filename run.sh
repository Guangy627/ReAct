#!/bin/bash
#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)
python examples/fever.py "$@"


#example
#./run.sh --logdir ./logs/fever --n 200 --checkpoint_every 50