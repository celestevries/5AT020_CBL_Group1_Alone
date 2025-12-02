from datetime import datetime

name = "Juan"
course = "5AT020"
now = datetime.now()


with open("test_slurm.txt", "w") as f:
    f.write(f"Name: {name}\nCourse: {course}\nNow: {now}\n")