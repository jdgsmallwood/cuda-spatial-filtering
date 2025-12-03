
## Running marimo

```
bash
source .venv/bin/activate
marimo edit
```

Take note of the port it's running on - somewhere around 2718

ssh to blackmesa and forward that port to your computer
ssh -L <port>:localhost<port>  blackmesa

for example:
```
ssh -L 2718:localhost:2718 blackmesa
```


