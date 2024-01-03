# Python
import subprocess  # For running commands
import json
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from pkg_resources import parse_requirements

# Custom
import manticore

def run_command(*args):
    return subprocess.check_output(args).decode("ascii").strip()

def git_hash():
    return run_command("git", "rev-parse", "HEAD")

def modules_dict(line_sep="\r\n"):
    all_modules = run_command("pip", "list", "--format=freeze")
    module_versions = dict(module.split("==", 1) for module in all_modules.split(line_sep))

    requirements = Path(manticore.__file__).parent.with_name("requirements.txt")

    out = {}
    with open(requirements) as f:
        for req in parse_requirements(f):
            out[req.name] = module_versions[req.name]

    return out

def timestamp():
    return datetime.now().strftime(f"%Y%m%d-%H%M%S")

class Experiment(ABC):
    def __init__(self, name):
        self.name = name

    def run_experiment(self):
        result = ExperimentResult(self)
        self.run(result.data)
        result.finish()
        return result

    @abstractmethod
    def run(self, data):
        pass

    @abstractmethod
    def to_json(self, path=None):
        pass

class ExperimentResult:
    def __init__(self, experiment):
        self.experiment = experiment
        self.start_time = timestamp()
        # self.git_hash = git_hash()
        # self.modules = modules_dict()
        self.data = dict()

    def finish(self):
        self.stop_time = timestamp()
    
    def to_json(self, path=None):
        out = dict()
        out["start_time"] = self.start_time
        out["stop_time"] = self.stop_time
        # out["git_hash"] = self.git_hash
        # out["modules"] = self.modules
        out["data"] = self.data

        path = path or f"results/{self.start_time}_{self.experiment.name}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=4)
        return path

    # def from_json(self):
        # with open(f"{name}.json") as f:
        #     out = json.load(f)
        #     print(out)
        #     print(type(out))

    def __repr__(self):
        return f"{self.__class__.__name__}(stop_time={self.stop_time}, data={self.data})" # git_hash={self.git_hash}, 

"""
MULTIPROCESSING NOTES

Why you shouldn't queue.get() after process.join()
https://stackoverflow.com/questions/31665328/python-3-multiprocessing-queue-deadlock-when-calling-join-before-the-queue-is-em

The subprocess will be blocked in put() waiting for the main process to remove some data from the queue with get(), but the main process is blocked in join() waiting for the subprocess to finish. This results in a deadlock.

Queue / Pipes should be used for arbitrary / two-way communication between subprocesses respectively.
"""

"""
Python child processes are instantiated with a copy of the parent's memory. On Linux machines, this copy is 'lazy' in the sense that child processes only perform the
copy whenever they modify the memory (Copy On Write). Writing to memory in children's processes will thus not be reflected in the parent's process, unless these values
are written to shared memory (e.g. mp.Array) or sent explicitly over Queues or Pipes.
"""
