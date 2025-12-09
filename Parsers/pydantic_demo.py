from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class Student(BaseModel):
    name: str = 'anuj'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10)

new_student = {'age':32,'email': 'abc@gmail.com', 'cgpa': 9.8}

student = Student(**new_student)

# print(student)

student_dict = dict(student)
print(student_dict)
