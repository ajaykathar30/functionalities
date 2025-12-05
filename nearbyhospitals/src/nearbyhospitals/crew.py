from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

# Import your custom tool
from .tools.custom_tool import MyCustomTool


@CrewBase
class Nearbyhospitals():
    """Nearbyhospitals crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Define hospital finder agent with tool attached
    @agent
    def hospital_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['hospital_finder'],  # ✅ use agents_config
            tools=[MyCustomTool()],
            verbose=True
        )
        
    @agent
    def hospital_reporter(self) -> Agent:
        return Agent(
        config=self.agents_config['hospital_reporter'],
        verbose=True
    )


    # Example research task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],  # type: ignore[index]
        )

    # Example reporting task
    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],  # type: ignore[index]
            output_file='report.md'
        )

    # Create the crew
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,   # auto-created from @agent
            tasks=self.tasks,     # auto-created from @task
            process=Process.sequential,
            verbose=True,
        )
