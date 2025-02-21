import subprocess

def get_pixmax(ID, cota, cotb, process):
    process.stdin.write(f"{ID} {cota} {cotb}\n")
    process.stdin.flush()
    
    output = process.stdout.readline()
    
    if not output:
        raise RuntimeError("No output from the process")
    
    return(output.strip())

def start_template_process(ID):
    return subprocess.Popen(
        ["./decapitation", str(ID), "0", "0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

ID = 1122
cota = 0.0
cotb = 0.0

process = start_template_process(ID)

for i in range(100):
    cota = -10.+i*0.1
    for j in range(10):
        cotb = -5 + j*0.5
        maxpix = get_pixmax(ID, cota, cotb, process)
        print("Max pixel value:", maxpix)

process.terminate()
process.wait()
