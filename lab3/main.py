from lab3.pipelines.first_task_pipeline import FirstTaskPipeline
from lab3.pipelines.second_task_pipeline import SecondTaskPipeline
from lab3.pipelines.third_task_pipeline import ThirdTaskPipeline


def main():
    file_path = "resources/weather.csv"

    #first_task = FirstTaskPipeline(file_path)
    #first_task.run()

    second_task = SecondTaskPipeline(file_path)
    second_task.run()

    #third_task = ThirdTaskPipeline(file_path)
    #third_task.run()


if __name__ == "__main__":
    main()
