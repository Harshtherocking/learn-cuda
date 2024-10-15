main : main.cu
	@mkdir -p ./bin
	@nvcc main.cu ./utils/mat.cu -o ./bin/temp
	@echo "successfully built"
	@./bin/temp

prof : main.cu
	@nvprof ./bin/temp

clean : 
	@rm -rf ./bin
	@clear
