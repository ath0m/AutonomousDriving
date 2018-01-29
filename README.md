# Autonomous Driving

Ten projeekt jest próbą stwrzenia autonomicznego samochodu w symulowanym środowisku. Moim poligonem będzie symulator [Carla](http://www.carla.org)

## Instalacja

System obecnie rozwijany jest pod Linux'em, a w szególność Manjaro

Po sklonowaniu należy pobrać symulator [Carla 0.7.1](https://drive.google.com/open?id=1IOh3HWQAU2kcdpyPxWfNaNAnwCaUKv2P) do folderu **CARLA**

```bash
$ virtualenv venv
$ . venv/bin/activate
$ cd CARLA/PythonClient
$ pip install -r requirements.txt
$ python setup.py install
```
