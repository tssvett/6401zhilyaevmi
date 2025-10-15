from lab3.pipelines.first_task_pipeline import FirstTaskPipeline
from lab3.pipelines.second_task_pipeline import SecondTaskPipeline
from lab3.pipelines.third_task_pipeline import ThirdTaskPipeline


def main():
    file_path = "resources/weather.csv"

    first_task = FirstTaskPipeline()
    first_task.run(file_path)

    second_task = SecondTaskPipeline()
    second_task.run(file_path)

    third_task = ThirdTaskPipeline()
    third_task.run(file_path)


if __name__ == "__main__":
    main()
