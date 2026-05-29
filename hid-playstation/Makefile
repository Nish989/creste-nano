obj-m += hid-playstation.o

KDIR := /lib/modules/$(shell uname -r)/build

all:
	make -C $(KDIR) M=$(PWD) modules

clean:
	make -C $(KDIR) M=$(PWD) clean

install:
	sudo cp hid-playstation.ko /lib/modules/$(shell uname -r)/kernel/drivers/hid/
	sudo depmod -a
