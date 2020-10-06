build:
	mkdir -p ./out/predict ./out/rect checkpoints

clean: 
	rm -rf ./checkpoints/*
	rm -rf ./out/rect/* ./out/predict/*
	find ./ -type d -name __pycache__ -exec rm -rf {} \;