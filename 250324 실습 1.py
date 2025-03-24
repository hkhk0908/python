class Book:
    def __init__(self, title, author, price):
        self.title = title
        self.author = author
        self.price = price

    def display_info(self):
        print(f"제목: {self.title}, 저자: {self.author}, 가격: {self.price}원")

    def __eq__(self, other):
        if isinstance(other, Book):
            return self.price == other.price
        return False


book1 = Book("파이썬 프로그래밍", "홍길동", 30000)
book2 = Book("알고리즘 기초", "이순신", 30000)
book3 = Book("자료구조", "강감찬", 25000)

book1.display_info()
book2.display_info()
book3.display_info()

print(book1 == book2)