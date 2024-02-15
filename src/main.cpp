#include <cassert>
#include <fstream>
#include <iostream>

struct Data {
    int value;
};

#pragma pack(1)
struct Other {
    int value;
    bool other;
};

template <typename T>
bool is_type(void *pool, void *ptr) {
    return ((uintptr_t)ptr % alignof(T)) == ((uintptr_t)pool % alignof(T));
}

int main(int argc, char **argv) {
    /*void *pool = malloc(sizeof(Data)+sizeof(Other));

    *((Data*)pool) = Data { 100 };
    *((Other*)(pool+sizeof(Data))) = Other { true };

    std::cout << "Pool index of 0 for Data " << is_type((Data*)pool) <<
    std::endl;
    //std::cout << "Pool index of 0 for Other" << is_type((Other*)pool) <<
    std::endl; std::cout << "Pool index of 1 for Other " <<
    is_type((Other*)(pool+sizeof(Data))) << std::endl;
    //std::cout << "Pool index of 1 for Data" <<
    is_type((Data*)(pool+sizeof(Data))) << std::endl;

    free(pool);*/

    Data *pool = new Data[3];

    Other *other_pool = new Other[4]+1;
    for (int i = 0; i < 3; i++) {
        other_pool[i] = Other{i, false};
    }

    for (int i = 0; i < 3; i++) {
        pool[i] = Data{i};
        std::cout << "Pool, index of " << i << " is: "
                  << is_type<Data>(pool, &pool[i]) /*<< " " << &pool[i]*/
                  << std::endl;
    }
    std::cout << "Pool, cmp to other: "
              << is_type<Data>(pool, &other_pool[0]) /*<< " " << other_pool*/
              << std::endl;
    std::cout << "alignof(Data) " << sizeof(Data) << std::endl;
    std::cout << "alignof(Other) " << sizeof(Other) << std::endl;

    delete pool;
}