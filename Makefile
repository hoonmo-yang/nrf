CURRENT_DIR = $(shell pwd)
BASHRC = ~/.bashrc
CENAI_DIR_EXISTS = $(shell grep -q 'export CENAI_DIR=' $(BASHRC) && echo "yes" || echo "no")

list::
	conda $@

install::
	@pip install -U -r requirements.txt
	@pip freeze > freeze.txt
	@if [ "$(CENAI_DIR_EXISTS)" = "yes" ]; then \
		sed -i "s|export CENAI_DIR=.*|export CENAI_DIR=\"$(CURRENT_DIR)\"|" $(BASHRC); \
		echo "envar 'CENAI_DIR' changed: $(CURRENT_DIR)"; \
	else \
		echo "export CENAI_DIR=\"$(CURRENT_DIR)\"" >> $(BASHRC); \
		echo "envar 'CENAI_DIR' added: $(CURRENT_DIR)"; \
	fi
	@echo "export PYTHONPATH=\$$PYTHONPATH:\$$CENAI_DIR" >> $(BASHRC)
	@echo "envar 'PYTHONPATH' added: $(CURRENT_DIR)"
	@echo "RUN command: source ~/.bashrc"

clean:: 
	pip uninstall -y -r requirements.txt
	pip freeze > freeze.txt
