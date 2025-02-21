import subprocess
class ThresholdManager:
    '''
    Deals with opening pixel templates and reading 
    the pixel charge values that we use as limits in clusters
    '''
    def __init__(self, ID):
        self.ID = ID
        self.process = self.start_template_process()

    def start_template_process(self):
        return subprocess.Popen(
            ["../CMSSW_templates/bin/decapitation", str(self.ID), "0", "0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def get_pixmax(self, cota, cotb):        
        self.process.stdin.write(f"{self.ID} {cota} {cotb}\n")
        self.process.stdin.flush()
        output = self.process.stdout.readline()

        if not output:
            raise RuntimeError("No output from the process")
        return output.strip()

    def terminate_process(self):
        self.process.terminate()
        self.process.wait()