#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import time
from datetime import timedelta

# Define URLs and output paths
urls = [
    "https://data.mendeley.com/public-files/datasets/5rr22hgzwr/files/8b0fe878-1a28-4adc-81ed-d9c146b93c79/file_downloaded",
    "https://data.mendeley.com/public-files/datasets/5rr22hgzwr/files/893301b3-8ab3-47b2-92eb-93fb426ff6ff/file_downloaded"
]

output_files = [
    "data/lung_cancer_test.pkl",
    "data/lung_cancer_train.pkl"
]

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def download_with_tqdm():
    """Download using requests with tqdm progress bar"""
    try:
        import requests
        from tqdm import tqdm
        
        for url, output_path in zip(urls, output_files):
            response = requests.get(url, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            progress_bar = tqdm(
                total=total_size, 
                unit='iB', 
                unit_scale=True,
                desc=os.path.basename(output_path)
            )
            
            with open(output_path, 'wb') as f:
                start_time = time.time()
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
                        
                        # Calculate and display speed and ETA
                        downloaded += len(chunk)
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            speed = downloaded / elapsed
                            eta = (total_size - downloaded) / speed if speed > 0 and total_size > 0 else 0
                            progress_bar.set_postfix(
                                speed=f"{speed/1024/1024:.2f} MB/s", 
                                eta=str(timedelta(seconds=int(eta)))
                            )
            
            progress_bar.close()
            print(f"Successfully downloaded to {output_path}")
        return True
    except ImportError:
        print("Python tqdm or requests library not available.")
        return False
    except Exception as e:
        print(f"Error with Python requests+tqdm: {e}")
        return False

def download_with_rich():
    """Download using requests with rich progress bar"""
    try:
        import requests
        from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
        
        for url, output_path in zip(urls, output_files):
            print(f"Downloading from {url} to {output_path}")
            
            # Make a request to get the file size
            response = requests.get(url, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            # Get total file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                DownloadColumn(),
                "•",
                TransferSpeedColumn(),
                "•",
                TimeRemainingColumn(),
            ) as progress:
                
                # Create a task for tracking
                task_id = progress.add_task(f"[green]Downloading {os.path.basename(output_path)}", total=total_size)
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))
            
            print(f"Successfully downloaded to {output_path}")
        return True
    except ImportError:
        print("Python rich or requests library not available.")
        return False
    except Exception as e:
        print(f"Error with rich progress: {e}")
        return False

def download_with_curl():
    """Try downloading using curl with progress bar"""
    try:
        for url, output_path in zip(urls, output_files):
            print(f"Downloading with curl from {url} to {output_path}")
            # Use curl's built-in progress bar
            result = subprocess.run(
                ["curl", "-L", "-#", "-o", output_path, url], 
                check=True
            )
            print(f"\nSuccessfully downloaded to {output_path}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Curl download failed: {e}")
        return False

def download_with_powershell():
    """Try downloading using PowerShell with progress"""
    try:
        for url, output_path in zip(urls, output_files):
            print(f"Downloading with PowerShell from {url} to {output_path}")
            # PowerShell script with progress reporting
            ps_script = f"""
            $ProgressPreference = 'Continue'
            $url = '{url}'
            $output = '{output_path}'
            
            $wc = New-Object System.Net.WebClient
            $wc.Headers.Add("User-Agent", "PowerShell Script")
            
            $totalLength = 0
            try {{
                $response = [System.Net.HttpWebRequest]::Create($url).GetResponse()
                $totalLength = $response.ContentLength
                $response.Close()
            }} catch {{
                Write-Host "Could not determine file size"
            }}
            
            $elapsed = [System.Diagnostics.Stopwatch]::StartNew()
            
            $wc.DownloadFileAsync($url, $output)
            
            $prevProgress = 0
            while ($wc.IsBusy) {{
                if ($totalLength -gt 0) {{
                    $percentComplete = [int](($wc.DownloadProgressChangedEventArgs.BytesReceived / $totalLength) * 100)
                    if ($percentComplete -gt $prevProgress) {{
                        $speed = ($wc.DownloadProgressChangedEventArgs.BytesReceived / 1MB) / $elapsed.Elapsed.TotalSeconds
                        Write-Progress -Activity "Downloading $($output)" -Status "$percentComplete% Complete" -PercentComplete $percentComplete -CurrentOperation "$([math]::Round($speed, 2)) MB/s"
                        $prevProgress = $percentComplete
                    }}
                }}
                Start-Sleep -Milliseconds 100
            }}
            
            Write-Progress -Activity "Downloading $($output)" -Completed
            """
            
            result = subprocess.run(["powershell", "-Command", ps_script], 
                                   check=True)
            print(f"Successfully downloaded to {output_path}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"PowerShell download failed: {e}")
        return False

def download_with_wget():
    """Try downloading using wget with progress bar"""
    try:
        for url, output_path in zip(urls, output_files):
            print(f"Downloading with wget from {url} to {output_path}")
            # Use wget with progress bar
            result = subprocess.run(
                ["wget", "--progress=bar:force", "-O", output_path, url], 
                check=True
            )
            print(f"Successfully downloaded to {output_path}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"Wget download failed: {e}")
        return False

if download_with_tqdm():
    pass
elif download_with_rich():
    pass
elif platform.system() == "Windows":
    if not download_with_powershell():
        if not download_with_curl():
            download_with_wget()
else:  # macOS or Linux
    if not download_with_curl():
        download_with_wget()

print("Download process completed.")
