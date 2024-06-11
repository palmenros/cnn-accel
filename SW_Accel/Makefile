all: cnnSolver

cnnSolver: cnnSolver.cpp model.cpp cnn.cpp model.h cnn.h CAccelDriver.cpp CAccelDriver.hpp CConvDriver.cpp CConvDriver.hpp
	g++ -O3 -Wall cnnSolver.cpp model.cpp cnn.cpp CAccelDriver.cpp CConvDriver.cpp -o cnnSolver -lm -lcma -lpthread

clean:
	rm -f cnnSolver

