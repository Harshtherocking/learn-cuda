main : main.cu
	@mkdir -p ./bin
	@nvcc main.cu --expt-relaxed-constexpr -diag-suppress 550,549 ../utils/guassian.cu ../utils/mat.cu -o ./bin/temp 
	@./bin/temp

clean :
	@rm -rf ./bin
	@clear
