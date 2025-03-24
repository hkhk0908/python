class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def display_info(self):
        print(f"이름: {self.name}, 급여: {self.salary}원")


class Manager:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        self.team_members = []

    def add_team_member(self, employee):
        if isinstance(employee, Employee):
            self.team_members.append(employee)
            print(f"{employee.name}님이 팀에 추가되었습니다.")
        else:
            print("올바른 Employee 객체를 추가하세요.")

    def display_team(self):
        print(f"{self.name} 매니저의 팀원 목록:")
        if not self.team_members:
            print("팀원이 없습니다.")
        else:
            for member in self.team_members:
                member.display_info()


emp1 = Employee("김철수", 5000000)
emp2 = Employee("이영희", 6000000)
manager = Manager("박민수", 8000000)

emp1.display_info()
emp2.display_info()

manager.add_team_member(emp1)
manager.add_team_member(emp2)
manager.display_team()
