class Worker:
    """Base class for workers"""

    def __init__(self):
        self.money = 0

    def take_salary(self, salary):

        self.money += salary

    def work(self, filename1, filename2, func):
        """function reads matrices from files, check correctness and print result """
        mat1 = list(map(lambda x: list(map(int, x.split())), open(filename1, 'r').readlines()))
        mat2 = list(map(lambda x: list(map(int, x.split())), open(filename2, 'r').readlines()))
        if len(mat2) == len(mat1) and all(map(lambda x, y: len(x) == len(y), mat1, mat2)):
            print(list(map(lambda x: list(map(lambda y, z: func(y, z), x[0], x[1])), zip(mat1, mat2))))
        else:
            raise Exception('Uncorrect matrices')


class Pupa(Worker):

    def do_work(self, filename1, filename2):

        self.work(filename1, filename2, lambda x, y: x + y)


class Lupa(Worker):

    def do_work(self, filename1, filename2):
        self.work(filename1, filename2, lambda x, y: x - y)


class Accountant:

    def give_salary(self, worker, money):
        if isinstance(worker, Worker):
            worker.take_salary(money)


# tests
if __name__ == '__main__':
    a = Accountant()
    p = Pupa()
    l = Lupa()
    a.give_salary(p, 100)
    a.give_salary(l, 200)
    p.do_work('test.txt', 'text.txt')
    l.do_work('test.txt', 'text.txt')
    print(p.money, l.money)