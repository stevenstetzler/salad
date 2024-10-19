# salad

Search for Asteroids through Line Analysis of Detection catalogs

# First time setup

```
$ source bin/setup.sh
$ load_lsst
$ python -m venv ./env
$ python -m pip install -r requirements.txt
$ python -m pip install --prefix ./env -e .
```

New shell:
```
$ source bin/setup.sh
$ load_env
```

Link a butler repository:
```
$ ln -s /path/to/butler/repo ./data
```

# Searching for asteroids

## Individual steps

```
$ source bin/setup.sh
$ load_lsst
$ salad images $REPO differenceExp images.pkl --collections DEEP/20190403/A0c --where "instrument='DECam' and detector=1" 
$ salad detection images.pkl catalog.pkl --threshold 5.0 
$ salad project catalog.pkl projection.pkl --velocity 0.1 0.5 --angle 120 240 --dx 10
$ salad hough projection.pkl hough.pkl --dx 10
$ salad find_clusters hough.pkl clusters.pkl --threshold 25
$ salad refine clusters.pkl refined.pkl
$ salad salad gather refined.pkl gathered.pkl --catalog catalog.pkl --threshold 1.0
$ salad deduplicate gathered.pkl deduplicated.pkl --images images.pkl --origin-threshold 5 --beta-threshold 1
$ salad cluster.filter deduplicated.pkl filtered.pkl --velocity 0.08 0.52 --angle 120 240 --min-points 15 
```

Individual steps can be pipelined to avoid writing to disk:
```
$ salad project catalog.pkl --velocity 0.1 0.5 --angle 120 240 --dx 10 | salad hough --dx 10 | salad find_clusters --threshold 25 | salad refine | salad gather --catalog catalog.pkl --threshold 1.0 | salad deduplicate --images images.pkl --origin-threshold 5 --beta-threshold 1 | salad cluster.filter /dev/stdin filtered.pkl --velocity 0.08 0.52 --angle 120 240 --min-points 15 
```

## Pipelined search

```
$ source bin/setup.sh
$ load_env
$ python $(which salad) pipeline.search ./search/ asteroid -J 24 --velocity 0.1 0.5 --detectors 1 --snrs 3.0 --filter-velocity 0.08 0.52 1> asteroid_search.log 2>&1 & disown
```

Full run:
```
# all collections with >30 images
$ collections="DEEP/20210506/A1j DEEP/20190708/B0a DEEP/20210507/A0i DEEP/20210507/A1f DEEP/20220823/B1m DEEP/20190828/B0a DEEP/20190829/B0b DEEP/20190827/B0c DEEP/20190403/A0c DEEP/20190402/A0b DEEP/20190601/A1b DEEP/20190827/B1c DEEP/20190828/B1a DEEP/20211004/B1c DEEP/20210515/A0h DEEP/20201015/B1b DEEP/20211003/B1i DEEP/20210513/A0e DEEP/20201019/B1d DEEP/20201021/B1d DEEP/20190829/B1b DEEP/20201020/B1f DEEP/20190603/A1c DEEP/20190926/B1a DEEP/20190927/B1b DEEP/20211005/B1h DEEP/20211006/B1e DEEP/20190602/A1a DEEP/20190928/B1c DEEP/20201016/B1c DEEP/20201017/B1e DEEP/20210515/A1h DEEP/20210516/A0a DEEP/20220825/B1e DEEP/20210503/A0j DEEP/20210928/B1g DEEP/20190505/A0c DEEP/20201018/B1a DEEP/20210509/A1f DEEP/20210927/B1d DEEP/20211002/B1f DEEP/20210510/A0c DEEP/20210516/A1a DEEP/20190504/A0a DEEP/20190507/A0b DEEP/20210930/B1j DEEP/20190507/A1b DEEP/20210513/A1c DEEP/20210513/A1e DEEP/20211001/B1b DEEP/20190505/A1c DEEP/20190707/B0a DEEP/20210516/A0b DEEP/20190504/A1a DEEP/20220525/A0a DEEP/20210910/B1a DEEP/20220527/A0d DEEP/20220526/A0b DEEP/20210904/B0a DEEP/20210908/B1h DEEP/20210912/B1b DEEP/20210906/B0f DEEP/20210509/A0g DEEP/20210903/B0b DEEP/20220821/B1a DEEP/20220827/B1i DEEP/20210909/B1a DEEP/20220826/B1k DEEP/20220821/B1h DEEP/20190706/B0a DEEP/20210903/B1f DEEP/20220526/A0c DEEP/20210510/A1f DEEP/20220826/B1l DEEP/20220827/B1b DEEP/20220525/A0f DEEP/20210904/B1e DEEP/20220527/A0e DEEP/20220822/B1g DEEP/20210518/A1d DEEP/20210907/B0d DEEP/20210504/A0f DEEP/20210906/B1c DEEP/20210905/B0c DEEP/20210905/B1i DEEP/20220822/B1d DEEP/20210912/B0e DEEP/20210506/A0i DEEP/20210908/B0d DEEP/20210910/B0a DEEP/20210909/B0c DEEP/20210510/A1i DEEP/20210907/B1h"

# scale out to 128*8=1024 cores
$ SALAD_SITE=klone SALAD_PARTITION=ckpt-all SALAD_ACCOUNT=astro-ckpt SALAD_CORES=8 SALAD_MEMORY=64 python $(which salad) pipeline.search ./search/ asteroid -J 128 --velocity 0.1 0.5 --collections ${collections} --filter-velocity 0.08 0.52 1> asteroid_search.log 2>&1 & disown
```


# Klone Configuration

```
$ export SALAD_SITE="klone"
```

