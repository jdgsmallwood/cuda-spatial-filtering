#!/bin/bash
tmux has-session -t cuda-spatial 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -s cuda-spatial
  # Optional: restore from last save
  tmux run-shell ~/.tmux/plugins/tmux-resurrect/scripts/restore.sh
else
  tmux attach -t cuda-spatial
fi
