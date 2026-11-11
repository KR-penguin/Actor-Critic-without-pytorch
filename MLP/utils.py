import numpy as np

def numerical_gradient(f, x, epsilon=1e-4):
    gradient = np.zeros_like(x) # x와 shape, data type(dtype)이 모두 동일하고, 모든 elements가 0인 배열 생성
    iterator = np.nditer(x, flags=['multi_index'])
    # np.editer (N-dimensional iterator)
    # flags=['multi_index'] option -> 현재 가리키고 있는 원소의 다차원 인덱스(튜플 형태)를 반환
    
    while not iterator.finished:
        index = iterator.multi_index
        origin = x[index]
        x[index] = origin + epsilon
        f_p = f()
        x[index] = origin - epsilon
        f_m = f()
        gradient[index] = (f_p - f_m) / (2*epsilon)
        x[index] = origin
        '''
        x는 mutable 객체라서 수정하면 x를 호출하는 f() 에도 반영됨
        자세한 내용은 test/참조전달.py 참고
        '''
        iterator.iternext()

    return gradient