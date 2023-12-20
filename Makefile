.PHONY: install_dvc
install_dvc:
	pip install dvc[ssh]==2.12.1 fsspec==2023.12.2


.PHONY: init_dvc
init_dvc:
	dvc init -f --no-scm
	dvc remote add --default my_remote ssh://91.206.15.25/home/a.gordeev/dvc_files
	dvc remote modify my_remote user a.gordeev
	dvc remote modify my_remote password $(STAGING_PASSWORD)
	dvc config cache.type hardlink,symlink


.PHONY: download_weights
download_weights:
	dvc pull -R weights
