.PHONY: install_dvc
install_dvc:
	pip install dvc[ssh]==3.34.0


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


.PHONY: deploy
deploy:
	ansible-playbook -i deploy/ansible/inventory.ini  deploy/ansible/deploy.yml \
		-e host=$(STAGING_HOST) \
		-e docker_image=$(DOCKER_IMAGE) \
		-e docker_tag=$(DOCKER_TAG) \
		-e docker_registry_user=$(CI_REGISTRY_USER) \
		-e docker_registry_password=$(CI_REGISTRY_PASSWORD) \
		-e docker_registry=$(CI_REGISTRY) \


.PHONY: destroy
destroy:
	ansible-playbook -i deploy/ansible/inventory.ini deploy/ansible/destroy.yml \
		-e host=$(DEPLOY_HOST)
