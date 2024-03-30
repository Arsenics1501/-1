.PHONY: prep txt compile grade clean

prep:
	make -C utils prep
	make -C utils

txt:
	make -C utils txt

compile:
	make -C src

grade:
	make -C src task2
	make -C src task3

clean:
	make -C src clean
	make -C utils clean
