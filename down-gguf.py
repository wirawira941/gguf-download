import requests
import json
import os
import shutil  # For shutil.which
import subprocess # For running external commands

NUM_CONNECTIONS_EXTERNAL = 8 # Default connections for axel/aria2c

def display_intro():
    print(r"""
  _____       _                           _   _  __ _
 / ____|     | |                         | | | |/ _| |
| |  __  ___ | |_   _ __ ___   __ _ _ __ | |_| | |_| | _____  __
| | |_ |/ _ \| | | | | '_ ` _ \ / _` | '_ \| __| |  _| |/ _ \ \/ /
| |__| | (_) | | |_| | | | | | | (_| | | | | |_| | | | |  __/>  <
 \_____|\___/|_|\__, |_| |_| |_|\__,_|_| |_|\__|_|_| |_|\___/_/\_\
                 __/ |
                |___/
    """)
    print("Welcome to the Ollama GGUF Model Downloader!")
    print("This script helps you download models directly from the Ollama library in GGUF format.\n")
    print("You'll need to provide:")
    print("- Model name (e.g., 'phi3')")
    print("- Model parameters/tag (e.g., 'mini', '3.8b', 'latest', 'medium:latest-q4_0')")
    print("\nLet's get started!\n")

def get_model_info():
    model_name = input("Enter the model name (e.g., 'phi3'): ").strip()
    model_params = input("Enter the model parameters/tag (e.g., 'mini', '3.8b-instruct-fp16'): ").strip()
    return model_name, model_params

def get_model_details(model_name, model_params):
    """Get both the manifest and metadata for the model"""
    manifest_url = f"https://registry.ollama.ai/v2/library/{model_name}/manifests/{model_params}"
    headers = {
        'User-Agent': 'GGUF-Downloader/1.0',
        'Accept': 'application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json'
    }

    try:
        response = requests.get(manifest_url, headers=headers, timeout=20)
        response.raise_for_status()
        manifest = response.json()

        model_digest = None
        config_data = {
            'model_family': model_name,
            'model_type': model_params,
            'file_type': 'unknown_quant'
        }

        for layer in manifest.get('layers', []):
            if layer.get('mediaType') == 'application/vnd.ollama.image.model':
                model_digest = layer['digest'].split(':')[-1]
                break

        if not model_digest and 'config' in manifest and 'digest' in manifest['config']:
            config_digest_full = manifest['config']['digest']
            config_url = f"https://registry.ollama.ai/v2/library/{model_name}/blobs/{config_digest_full}"
            config_response = requests.get(config_url, headers=headers, timeout=20)
            config_response.raise_for_status()
            try:
                temp_config_data = config_response.json()
                if isinstance(temp_config_data, dict):
                    config_data['model_family'] = temp_config_data.get('model_family', config_data['model_family'])
                    config_data['model_type'] = temp_config_data.get('model_type', config_data['model_type'])
                    config_data['file_type'] = temp_config_data.get('file_type', config_data['file_type'])
                    general_params = temp_config_data.get('general', {})
                    if 'quantization' in general_params and (config_data['file_type'] == 'unknown_quant' or not config_data['file_type']):
                        config_data['file_type'] = general_params['quantization']
                if 'rootfs' in temp_config_data and 'diff_ids' in temp_config_data['rootfs'] and temp_config_data['rootfs']['diff_ids']:
                    model_digest = temp_config_data['rootfs']['diff_ids'][0].split(':')[-1]
            except json.JSONDecodeError:
                print(f"Warning: Could not parse config blob from {config_url} as JSON.")
            except requests.exceptions.RequestException as e_config:
                 print(f"Warning: Failed to fetch config blob from {config_url}: {str(e_config)}.")

        if not model_digest and 'layers' in manifest and manifest['layers']:
            if 'digest' in manifest['layers'][0]:
                model_digest = manifest['layers'][0]['digest'].split(':')[-1]
                print("Warning: Using digest of the first layer from manifest as fallback.")
            else:
                raise ValueError("Could not find model data in manifest; first layer has no digest.")

        if not model_digest:
            raise ValueError("Could not determine model data digest from the manifest.")

        if config_data['file_type'] == 'unknown_quant' or not config_data['file_type']:
            tag_lower = model_params.lower()
            quant_patterns = {
                'q4_0': ['q4_0'], 'q4_1': ['q4_1'], 'q5_0': ['q5_0'], 'q5_1': ['q5_1'],
                'q8_0': ['q8_0'], 'q2_k': ['q2_k'], 'q3_k_s': ['q3_k_s'], 'q3_k_m': ['q3_k_m'],
                'q3_k_l': ['q3_k_l'], 'q4_k_s': ['q4_k_s'], 'q4_k_m': ['q4_k_m'], 'q5_k_s': ['q5_k_s'],
                'q5_k_m': ['q5_k_m'], 'q6_k': ['q6_k'], 'fp16': ['f16', 'fp16']
            }
            found_quant_in_tag = False
            for quant, patterns_list in quant_patterns.items():
                for p in patterns_list:
                    if p in tag_lower:
                        config_data['file_type'] = quant.upper()
                        found_quant_in_tag = True
                        break
                if found_quant_in_tag:
                    break
            if not found_quant_in_tag:
                config_data['file_type'] = 'Q4_0'
                print(f"Warning: Defaulting quantization to '{config_data['file_type']}'.")

        return {
            'digest': model_digest,
            'model_family': config_data['model_family'],
            'model_type': config_data['model_type'],
            'file_type': config_data['file_type']
        }
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
            raise Exception(f"Failed to fetch model details for '{model_name}:{model_params}'. Model or tag not found (404). URL: {manifest_url}")
        raise Exception(f"Failed to fetch model details: {str(e)}. URL: {manifest_url}")
    except json.JSONDecodeError:
        raise Exception(f"Failed to parse manifest response as JSON. URL: {manifest_url}")
    except (KeyError, IndexError, ValueError) as e:
        raise Exception(f"Failed to parse manifest structure or find digest. Error: {str(e)}")

def check_downloader_availability():
    """Checks for axel and aria2c and returns their paths if found."""
    available = {}
    axel_path = shutil.which("axel")
    if axel_path:
        available["axel"] = axel_path
    aria2_path = shutil.which("aria2c")
    if aria2_path:
        available["aria2c"] = aria2_path
    return available

def select_download_manager(available_downloaders):
    """Allows user to select a download manager."""
    print("\n🚀 Select Download Manager:")
    options = {"1": ("requests", "Built-in Python requests (single connection)")}
    idx = 2
    if "axel" in available_downloaders:
        options[str(idx)] = ("axel", f"axel (multi-connection, path: {available_downloaders['axel']})")
        idx += 1
    if "aria2c" in available_downloaders:
        options[str(idx)] = ("aria2c", f"aria2c (multi-connection, path: {available_downloaders['aria2c']})")
        idx +=1

    if len(options) == 1:
        print("Using built-in Python requests downloader.")
        return "requests"

    for key, (name, desc) in options.items():
        print(f"  {key}. {desc}")

    while True:
        choice = input(f"Enter your choice (1-{len(options)}, default 1 for requests): ").strip() or "1"
        if choice in options:
            return options[choice][0]
        else:
            print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")

def download_model(model_name, model_details, filename, downloader_choice="requests", user_confirmed_overwrite=False):
    digest_hash = model_details['digest']
    blob_digest_arg = f"sha256:{digest_hash}" if not digest_hash.startswith("sha256:") else digest_hash
    url = f"https://registry.ollama.ai/v2/library/{model_name}/blobs/{blob_digest_arg}"

    print(f"\nAttempting to download using: {downloader_choice}")
    print(f"Downloading from: {url}")

    if downloader_choice == "axel":
        cmd = ["axel", "-n", str(NUM_CONNECTIONS_EXTERNAL), "-a", "-o", filename, url]
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True) # Allows live output from axel
            print(f"\n✅ Download complete using axel! Saved as {filename}")
            return True
        except FileNotFoundError:
            raise Exception("axel command not found. Please ensure it's installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            if os.path.exists(filename): try: os.remove(filename) catch OSError: pass
            raise Exception(f"axel download failed with exit code {e.returncode}. Check axel's output above.")

    elif downloader_choice == "aria2c":
        cmd = [
            "aria2c",
            "-x", str(NUM_CONNECTIONS_EXTERNAL),
            "-s", str(NUM_CONNECTIONS_EXTERNAL),
            "--console-log-level=warn",
            "-d", os.path.dirname(filename) or ".",
            "-o", os.path.basename(filename)
        ]
        if user_confirmed_overwrite:
            cmd.append("--allow-overwrite=true")
        cmd.append(url) # URL must be the last argument for aria2c typically

        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True) # Allows live output from aria2c
            print(f"\n✅ Download complete using aria2c! Saved as {filename}")
            return True
        except FileNotFoundError:
            raise Exception("aria2c command not found. Please ensure it's installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            if os.path.exists(filename) and e.returncode !=0 : try: os.remove(filename) catch OSError: pass
            raise Exception(f"aria2c download failed with exit code {e.returncode}. Check aria2c's output above.")

    elif downloader_choice == "requests":
        headers_req = {'User-Agent': 'GGUF-Downloader/1.0', 'Accept': 'application/octet-stream'}
        try:
            with requests.get(url, headers=headers_req, stream=True, timeout=(20, None)) as response:
                response.raise_for_status()
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    try:
                        error_data = response.json()
                        error_message = error_data.get('errors', [{}])[0].get('message', json.dumps(error_data))
                        raise Exception(f"Server returned JSON error: {error_message} from {url}")
                    except (json.JSONDecodeError, IndexError, KeyError):
                        raise Exception(f"Server returned JSON instead of binary (couldn't parse error) from {url}")

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                if total_size > 0: print(f"Total size: {total_size / (1024 * 1024):.2f} MB")

                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192 * 4):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"Downloaded: {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent:.1f}%)", end='\r')
                            else:
                                print(f"Downloaded: {downloaded / (1024*1024):.2f} MB (total size unknown)", end='\r')
                if total_size > 0: print(" " * 70, end='\r') # Clear progress line
                print(f"\n✅ Download complete using requests! Saved as {filename}")
                return True
        except requests.exceptions.RequestException as e_req:
            if os.path.exists(filename): try: os.remove(filename) except OSError: pass
            if hasattr(e_req, 'response') and e_req.response is not None and e_req.response.status_code == 404:
                raise Exception(f"Requests download failed: Blob not found (404) at {url}.")
            raise Exception(f"Requests download failed: {str(e_req)}. URL: {url}")
    else:
        raise ValueError(f"Unknown downloader choice: {downloader_choice}")


def main():
    display_intro()
    downloader_choice = "requests" # Default

    try:
        model_name, model_params = get_model_info()
        if not model_name or not model_params:
            print("❌ Model name and parameters cannot be empty. Exiting.")
            return

        print("\n🛠  Fetching model information...")
        model_details = get_model_details(model_name, model_params)
        
        print(f"  Model Family: {model_details['model_family']}")
        print(f"  Model Type/Tag: {model_details['model_type']}")
        print(f"  File Type (Quantization): {model_details['file_type']}")
        print(f"  Blob Digest: {model_details['digest'][:12]}...")

        available_external_downloaders = check_downloader_availability()
        if available_external_downloaders:
            downloader_choice = select_download_manager(available_external_downloaders)
        else:
            print("\nNo external download managers (axel, aria2c) found. Using built-in Python requests.")

        safe_model_params = model_params.replace(":", "-").replace("/", "-")
        default_filename = f"{model_name}-{safe_model_params}-{model_details['file_type']}.gguf"
        
        filename_input = input(f"\n📝 Enter output filename (default: {default_filename}): ").strip()
        filename = filename_input or default_filename
        filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', '.'))
        if not filename.endswith(".gguf"): filename += ".gguf"

        user_confirmed_overwrite = False
        if os.path.exists(filename):
            overwrite_choice = input(f"⚠️ File '{filename}' already exists. Overwrite? (y/N): ").strip().lower()
            if overwrite_choice == 'y':
                user_confirmed_overwrite = True
                if downloader_choice == "axel": # Axel might create filename.1, so remove original first
                    try:
                        os.remove(filename)
                        print(f"Removed existing file '{filename}' to ensure axel overwrites correctly.")
                    except OSError as e:
                        print(f"Warning: Could not remove existing file '{filename}': {e}. Axel might create a new file like '{filename}.1'.")
            else:
                print("Download cancelled by user (file exists).")
                return
        
        download_model(model_name, model_details, filename, downloader_choice, user_confirmed_overwrite)

        print("\n🎉 All done! Happy AI experimenting!")
        print(f"You can now use '{filename}' with llama.cpp based tools.")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPossible reasons:")
        print("- Incorrect model name or parameters/tag.")
        print("- Network connection issues or Ollama registry temporarily unavailable.")
        print("- Changes in Ollama's API structure.")
        print("- The specific model tag might not resolve to a GGUF file or chosen downloader failed.")
        print("\nPlease check your inputs, network, and try again.")

if __name__ == "__main__":
    main()
