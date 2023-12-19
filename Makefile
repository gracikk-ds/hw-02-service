.PHONY: install_dvc
install_dvc:
	pip install pygit2==1.10.1 pathspec==0.9.0
	pip install dvc[ssh]==2.5.4


.PHONY: init_dvc
init_dvc:
	dvc init --no-scm
	dvc remote add --default my_remote ssh://91.206.15.25/home/$(STAGING_USERNAME)/dvc_files
	dvc remote modify my_remote user $(STAGING_USERNAME) password $(STAGING_PASSWORD)
	dvc config cache.type hardlink,symlink


.PHONY: download_weights
download_weights:
	dvc pull -R weights
