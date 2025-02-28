#!/usr/bin/env python3
"""
Fast Model Downloader - Optimized for Azure A100 GPU Servers

This script efficiently downloads large AI models from URLs or Hugging Face repositories,
taking full advantage of Azure's networking capabilities and A100 server performance.

Features:
- Multi-part parallel downloading with configurable chunks
- Automatic resume of interrupted downloads
- Hugging Face repository support with authentication
- Direct URL downloads with optimized TCP settings
- Detailed progress tracking with ETA
- SHA256 checksum verification
- Optimized for Azure datacenter networking
"""

import os
import sys
import asyncio
import logging
import hashlib
import argparse
import json
import time
import shutil
import tempfile
import math
from typing import Dict, List, Optional, Union, Tuple, Set, Any
from pathlib import Path
from urllib.parse import urlparse, unquote
from datetime import datetime, timedelta

import aiohttp
import aiofiles
import tqdm
import huggingface_hub
from huggingface_hub import snapshot_download, hf_hub_download, HfApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("model-downloader")

# Constants
DEFAULT_CONFIG_FILE = "model_config.json"
DEFAULT_OUTPUT_DIR = "models"
CHUNK_SIZE = 64 * 1024 * 1024  # 64MB chunks for parallel downloading
BUFFER_SIZE = 1024 * 1024  # 1MB buffer for file I/O
MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds
MAX_PARALLEL_DOWNLOADS = 4
MAX_PARALLEL_CHUNKS = 16
CONNECT_TIMEOUT = 30  # seconds
AZURE_DOWNLOAD_REGIONS = ["eastus", "westus2", "westeurope"]  # Prioritized Azure regions
HF_MIRROR_URL = "https://huggingface.co"  # Default HF mirror, can be changed for better region access


class DownloadManager:
    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        parallel_downloads: int = MAX_PARALLEL_DOWNLOADS,
        parallel_chunks: int = MAX_PARALLEL_CHUNKS,
        hf_token: Optional[str] = None,
        verify_checksums: bool = True,
        force_redownload: bool = False,
        azure_region: Optional[str] = None,
        tcp_optimization: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parallel_downloads = parallel_downloads
        self.parallel_chunks = parallel_chunks
        self.hf_token = hf_token
        self.verify_checksums = verify_checksums
        self.force_redownload = force_redownload
        self.azure_region = azure_region
        self.tcp_optimization = tcp_optimization
        self.semaphore = asyncio.Semaphore(parallel_downloads)
        self.already_downloaded: Set[str] = set()
        self.http_headers = {
            "User-Agent": "FastModelDownloader/1.0",
        }

        # Set up HF API with token if provided
        if hf_token:
            huggingface_hub.login(token=hf_token)
            self.hf_api = HfApi(token=hf_token)
            self.http_headers["Authorization"] = f"Bearer {hf_token}"

    async def fetch_with_progress(self, url: str, path: Path, file_size: Optional[int] = None, desc: Optional[str] = None) -> bool:
        """Fetch a URL with progress reporting"""
        start_time = time.time()
        desc = desc or f"Downloading {path.name}"

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use a temporary file for downloading
        temp_file = path.with_suffix(f"{path.suffix}.download")

        # Calculate existing size for resume
        existing_size = 0
        if temp_file.exists():
            existing_size = temp_file.stat().st_size
            logger.info(f"Found existing partial download: {temp_file} ({existing_size} bytes)")

        # Add Range header for resume
        headers = dict(self.http_headers)
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            logger.info(f"Resuming download from byte {existing_size}")

        try:
            # Use custom TCP settings for Azure optimization
            tcp_conn = None
            if self.tcp_optimization:
                tcp_conn = aiohttp.TCPConnector(
                    ssl=False,
                    force_close=False,
                    limit=self.parallel_chunks,
                    ttl_dns_cache=300,
                    family=4,  # IPv4 often has better performance on Azure
                    use_dns_cache=True,
                    limit_per_host=self.parallel_chunks // 2,
                )

            timeout = aiohttp.ClientTimeout(total=None, connect=CONNECT_TIMEOUT)

            async with aiohttp.ClientSession(headers=headers, connector=tcp_conn, timeout=timeout) as session:
                async with session.get(url) as response:
                    # Check for successful response
                    if response.status not in (200, 206):  # OK or Partial Content
                        logger.error(f"Failed to download {url}: HTTP {response.status}")
                        if temp_file.exists():
                            temp_file.unlink()
                        return False

                    # Get file size if not provided
                    if file_size is None:
                        content_length = response.headers.get("Content-Length")
                        if content_length:
                            file_size = int(content_length)
                        elif response.status == 206:
                            # For resumed downloads, get content range
                            content_range = response.headers.get("Content-Range", "")
                            if content_range:
                                try:
                                    file_size = int(content_range.split("/")[1])
                                except (IndexError, ValueError):
                                    file_size = None

                    if file_size and existing_size > 0:
                        total_size = file_size
                    else:
                        total_size = file_size or None

                    # Set up progress bar
                    progress = tqdm.tqdm(
                        desc=desc,
                        initial=existing_size,
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        dynamic_ncols=True,
                    )

                    # Download with progress
                    mode = "ab" if existing_size > 0 else "wb"

                    async with aiofiles.open(temp_file, mode) as f:
                        chunk_size = 1024 * 1024  # 1MB chunks for streaming
                        downloaded = existing_size

                        async for chunk in response.content.iter_chunked(chunk_size):
                            if chunk:
                                await f.write(chunk)
                                downloaded += len(chunk)
                                progress.update(len(chunk))

                                # Calculate and display speed and ETA
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    speed = downloaded / elapsed
                                    if total_size:
                                        eta = (total_size - downloaded) / speed if speed > 0 else 0
                                        progress.set_postfix({"speed": f"{speed / 1024 / 1024:.2f} MB/s", "eta": str(timedelta(seconds=int(eta)))})
                                    else:
                                        progress.set_postfix({"speed": f"{speed / 1024 / 1024:.2f} MB/s"})

                    progress.close()

                    # Check if download is complete
                    if total_size and downloaded < total_size:
                        logger.warning(f"Download incomplete: {downloaded}/{total_size} bytes")
                        return False

                    # Rename temp file to actual file
                    temp_file.rename(path)
                    logger.info(f"Download complete: {path}")
                    return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False

    async def download_file_parallel(self, url: str, path: Path, file_size: Optional[int] = None, desc: Optional[str] = None) -> bool:
        """Download a file in parallel chunks for better performance"""
        start_time = time.time()
        desc = desc or f"Downloading {path.name}"

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get file size first
        if file_size is None:
            async with aiohttp.ClientSession(headers=self.http_headers) as session:
                try:
                    async with session.head(url, allow_redirects=True) as response:
                        if response.status != 200:
                            # Try GET instead
                            async with session.get(url, allow_redirects=True) as get_response:
                                if "Content-Length" in get_response.headers:
                                    file_size = int(get_response.headers["Content-Length"])
                        elif "Content-Length" in response.headers:
                            file_size = int(response.headers["Content-Length"])
                except Exception as e:
                    logger.warning(f"Error getting file size for {url}: {str(e)}")

        # If file size is very small or not known, use simple download
        if file_size is None or file_size < 100 * 1024 * 1024:  # Less than 100MB
            return await self.fetch_with_progress(url, path, file_size, desc)

        # Use a temp directory for chunks
        temp_dir = Path(tempfile.mkdtemp(prefix=f"dl_{path.stem}_"))
        try:
            # Plan the chunks
            num_chunks = min(self.parallel_chunks, math.ceil(file_size / CHUNK_SIZE))
            chunk_size = math.ceil(file_size / num_chunks)

            logger.info(f"Downloading {path.name} in {num_chunks} parallel chunks of ~{chunk_size / (1024 * 1024):.1f}MB each")

            # Create a master progress bar
            master_progress = tqdm.tqdm(
                desc=desc,
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
            )

            # Function to download a specific chunk
            async def download_chunk(chunk_id, start, end):
                chunk_path = temp_dir / f"chunk_{chunk_id:05d}"

                # Skip if chunk already fully downloaded
                if chunk_path.exists() and chunk_path.stat().st_size == (end - start):
                    master_progress.update(end - start)
                    return True

                # Add range header for this chunk
                headers = dict(self.http_headers)
                headers["Range"] = f"bytes={start}-{end - 1}"

                retries = 0
                while retries < MAX_RETRIES:
                    try:
                        async with aiohttp.ClientSession(headers=headers) as session:
                            async with session.get(url) as response:
                                if response.status != 206:  # Partial Content
                                    logger.warning(f"Chunk {chunk_id} received non-206 status: {response.status}")
                                    if response.status not in (200, 416):  # OK or Range Not Satisfiable
                                        retries += 1
                                        await asyncio.sleep(RETRY_DELAY * retries)
                                        continue

                                async with aiofiles.open(chunk_path, "wb") as f:
                                    downloaded = 0
                                    async for data in response.content.iter_chunked(BUFFER_SIZE):
                                        await f.write(data)
                                        downloaded += len(data)
                                        master_progress.update(len(data))

                                # Verify chunk size
                                expected_size = min(chunk_size, end - start)
                                if chunk_path.stat().st_size != expected_size:
                                    logger.warning(f"Chunk {chunk_id} incomplete, retrying... ({chunk_path.stat().st_size}/{expected_size})")
                                    retries += 1
                                    await asyncio.sleep(RETRY_DELAY * retries)
                                    continue

                                return True
                    except Exception as e:
                        logger.warning(f"Error downloading chunk {chunk_id}: {str(e)}")
                        retries += 1
                        await asyncio.sleep(RETRY_DELAY * retries)

                logger.error(f"Failed to download chunk {chunk_id} after {MAX_RETRIES} retries")
                return False

            # Create download tasks for each chunk
            tasks = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, file_size)
                tasks.append(download_chunk(i, start, end))

            # Wait for all chunks to download
            results = await asyncio.gather(*tasks)
            master_progress.close()

            # Check if all chunks were downloaded
            if not all(results):
                logger.error(f"Some chunks failed to download for {path.name}")
                return False

            # Combine chunks into final file
            logger.info(f"Combining chunks for {path.name}")
            combine_progress = tqdm.tqdm(
                desc=f"Combining {path.name}",
                total=file_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            )

            async with aiofiles.open(path, "wb") as outfile:
                for i in range(num_chunks):
                    chunk_path = temp_dir / f"chunk_{i:05d}"
                    if not chunk_path.exists():
                        logger.error(f"Missing chunk {i} for {path.name}")
                        return False

                    # Read and write in buffered chunks to avoid loading entire file into memory
                    async with aiofiles.open(chunk_path, "rb") as infile:
                        while True:
                            chunk = await infile.read(BUFFER_SIZE)
                            if not chunk:
                                break
                            await outfile.write(chunk)
                            combine_progress.update(len(chunk))

            combine_progress.close()

            # Verify file size
            actual_size = path.stat().st_size
            if actual_size != file_size:
                logger.error(f"Combined file size mismatch: {actual_size} != {file_size}")
                return False

            logger.info(f"Successfully downloaded {path.name}")
            return True

        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory: {str(e)}")

    async def download_from_url(self, url: str, output_path: Path, desc: Optional[str] = None) -> bool:
        """Download a file from a direct URL"""
        # Check if already downloaded
        if output_path.exists() and not self.force_redownload:
            if output_path.stat().st_size > 0:
                logger.info(f"File already exists: {output_path}")
                self.already_downloaded.add(str(output_path))
                return True

        # Make sure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Acquire semaphore to limit parallel downloads
        async with self.semaphore:
            logger.info(f"Starting download: {url} -> {output_path}")
            if output_path.suffix.lower() in (".gguf", ".bin") and url.startswith("http"):
                # For large model files, use parallel chunk downloading
                return await self.download_file_parallel(url, output_path, desc=desc)
            else:
                # For smaller files, use simple progressive download
                return await self.fetch_with_progress(url, output_path, desc=desc)

    async def download_from_huggingface(self, repo_id: str, filename: Optional[str] = None, output_path: Optional[Path] = None) -> bool:
        """Download a file or repository from Hugging Face"""
        try:
            # Determine if we're downloading a specific file or an entire repo
            if filename:
                # Download a specific file
                if output_path is None:
                    output_path = self.output_dir / filename

                # Check if already downloaded
                if output_path.exists() and not self.force_redownload:
                    if output_path.stat().st_size > 0:
                        logger.info(f"File already exists: {output_path}")
                        self.already_downloaded.add(str(output_path))
                        return True

                logger.info(f"Downloading from Hugging Face: {repo_id}/{filename} -> {output_path}")

                # Check file size from HF API
                file_info = None
                try:
                    if hasattr(self, "hf_api"):
                        file_info = self.hf_api.model_info(repo_id, files_metadata=True)
                        for file in file_info.siblings:
                            if file.rfilename == filename:
                                file_size = file.size
                                break
                except:
                    file_size = None

                # For direct file download, get the download URL and use our parallel downloader
                file_url = huggingface_hub.hf_hub_url(repo_id, filename, revision="main")

                # Use our parallel downloader for better performance
                return await self.download_from_url(file_url, output_path, desc=f"Downloading {repo_id}/{filename}")
            else:
                # Download entire repository
                output_dir = output_path or self.output_dir / repo_id.split("/")[-1]
                logger.info(f"Downloading entire repository: {repo_id} -> {output_dir}")

                # Use snapshot_download from huggingface_hub
                # This runs synchronously, so we'll run it in a thread to not block
                loop = asyncio.get_event_loop()

                def download_repo():
                    return snapshot_download(
                        repo_id,
                        local_dir=output_dir,
                        token=self.hf_token,
                        revision="main",
                        ignore_patterns=[".*", "*.md", "*.txt"],
                    )

                try:
                    result = await loop.run_in_executor(None, download_repo)
                    logger.info(f"Successfully downloaded repository: {repo_id} -> {result}")
                    return True
                except Exception as e:
                    logger.error(f"Error downloading repository {repo_id}: {str(e)}")
                    return False

        except Exception as e:
            logger.error(f"Error downloading from Hugging Face: {str(e)}")
            return False

    async def process_model_config(self, config_file: str) -> List[Dict[str, Any]]:
        """Process the model config file and extract download information"""
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            # Extract model information
            download_tasks = []

            for model_type, model_config in config.items():
                model_path = model_config.get("model_path")
                model_name = model_config.get("model_name")
                model_url = model_config.get("model_url")
                model_type_str = model_config.get("model_type", "huggingface")
                download_if_missing = model_config.get("download_if_missing", True)

                if not download_if_missing:
                    logger.info(f"Skipping {model_type} model download as download_if_missing is False")
                    continue

                if not model_path:
                    logger.warning(f"No model_path specified for {model_type}, skipping")
                    continue

                output_path = self.output_dir / model_path

                # Add to download tasks
                task = {"model_type": model_type, "model_name": model_name, "model_path": model_path, "output_path": output_path}

                if model_url:
                    # Direct URL download
                    task["source"] = "url"
                    task["url"] = model_url
                elif model_name and "/" in model_name:
                    # Hugging Face repo
                    task["source"] = "huggingface"
                    if model_type_str.lower() == "huggingface":
                        # Download the entire repo
                        task["repo_id"] = model_name
                        task["filename"] = None
                    else:
                        # Just download the specific file
                        task["repo_id"] = "/".join(model_name.split("/")[:-1])
                        task["filename"] = model_name.split("/")[-1]
                else:
                    logger.warning(f"No valid source for {model_type} model, skipping")
                    continue

                download_tasks.append(task)

            return download_tasks
        except Exception as e:
            logger.error(f"Error processing config file: {str(e)}")
            return []

    async def download_task(self, task: Dict[str, Any]) -> bool:
        """Process a single download task"""
        try:
            model_type = task["model_type"]
            output_path = task["output_path"]

            logger.info(f"Processing download task for {model_type} model")

            if task["source"] == "url":
                return await self.download_from_url(task["url"], output_path, desc=f"Downloading {model_type} model")
            elif task["source"] == "huggingface":
                return await self.download_from_huggingface(task["repo_id"], task["filename"], output_path)
            else:
                logger.warning(f"Unknown source type: {task['source']}")
                return False
        except Exception as e:
            logger.error(f"Error in download task: {str(e)}")
            return False

    async def verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify the checksum of a downloaded file"""
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False

        # Determine checksum algorithm from length
        algorithm = "sha256"  # Default
        if len(expected_checksum) == 32:
            algorithm = "md5"
        elif len(expected_checksum) == 40:
            algorithm = "sha1"
        elif len(expected_checksum) == 64:
            algorithm = "sha256"
        elif len(expected_checksum) == 128:
            algorithm = "sha512"

        # Calculate checksum
        logger.info(f"Verifying {algorithm} checksum for {file_path.name}")
        hash_obj = getattr(hashlib, algorithm)()

        chunk_size = BUFFER_SIZE  # 1MB chunks
        total_size = file_path.stat().st_size

        with tqdm.tqdm(
            desc=f"Verifying {algorithm}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            async with aiofiles.open(file_path, "rb") as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    hash_obj.update(chunk)
                    progress.update(len(chunk))

        calculated_checksum = hash_obj.hexdigest()

        # Compare checksums
        if calculated_checksum.lower() == expected_checksum.lower():
            logger.info(f"Checksum verified for {file_path.name}")
            return True
        else:
            logger.error(f"Checksum mismatch for {file_path.name}")
            logger.error(f"Expected: {expected_checksum}")
            logger.error(f"Calculated: {calculated_checksum}")
            return False

    async def run(self, config_file: str) -> bool:
        """Run the download manager"""
        # Process config file
        download_tasks = await self.process_model_config(config_file)

        if not download_tasks:
            logger.warning("No valid download tasks found in config")
            return False

        # Execute downloads
        results = []
        for task in download_tasks:
            result = await self.download_task(task)
            results.append(result)

        # Report results
        success_count = sum(1 for r in results if r)
        total_count = len(results)

        if success_count == total_count:
            logger.info(f"Successfully downloaded all {total_count} models")
            return True
        else:
            logger.warning(f"Downloaded {success_count}/{total_count} models")
            return False


async def optimize_system():
    """Apply system optimizations for faster downloads on Azure"""
    try:
        # Set TCP window size (requires root)
        if os.geteuid() == 0:
            try:
                # TCP optimizations
                os.system("sysctl -w net.core.rmem_max=16777216")  # 16MB
                os.system("sysctl -w net.core.wmem_max=16777216")  # 16MB
                os.system('sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"')
                os.system('sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"')
                os.system("sysctl -w net.ipv4.tcp_window_scaling=1")
                os.system("sysctl -w net.ipv4.tcp_timestamps=1")
                os.system("sysctl -w net.ipv4.tcp_sack=1")

                # Higher backlog for busy servers
                os.system("sysctl -w net.core.netdev_max_backlog=5000")

                # Optimize for throughput
                os.system("sysctl -w net.ipv4.tcp_slow_start_after_idle=0")
                os.system("sysctl -w net.ipv4.tcp_mtu_probing=1")

                # Increase timeout values
                os.system("sysctl -w net.ipv4.tcp_keepalive_time=60")
                os.system("sysctl -w net.ipv4.tcp_keepalive_intvl=10")
                os.system("sysctl -w net.ipv4.tcp_keepalive_probes=6")

                # Avoid congestion on Azure network
                os.system("sysctl -w net.ipv4.tcp_congestion_control=bbr")

                logger.info("Applied system-level TCP optimizations")
            except Exception as e:
                logger.warning(f"Failed to apply system optimizations: {str(e)}")
        else:
            logger.info("Skipping system optimizations (requires root privileges)")

        # Set process I/O priority
        try:
            import psutil

            process = psutil.Process(os.getpid())
            process.ionice(psutil.IOPRIO_CLASS_RT)
            logger.info("Set I/O priority to real-time for current process")
        except (ImportError, Exception) as e:
            logger.warning(f"Could not set I/O priority: {str(e)}")

    except Exception as e:
        logger.warning(f"Error in system optimization: {str(e)}")


async def main():
    parser = argparse.ArgumentParser(description="Fast Model Downloader - Optimized for Azure A100 GPU Servers")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE, help="Path to model config file")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for downloaded models")
    parser.add_argument("--parallel-downloads", type=int, default=MAX_PARALLEL_DOWNLOADS, help="Maximum parallel downloads")
    parser.add_argument("--parallel-chunks", type=int, default=MAX_PARALLEL_CHUNKS, help="Maximum parallel chunks per download")
    parser.add_argument("--hf-token", type=str, help="Hugging Face API token")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--azure-region", type=str, choices=AZURE_DOWNLOAD_REGIONS, help="Prioritize Azure region")
    parser.add_argument("--skip-optimizations", action="store_true", help="Skip TCP and system optimizations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--model", type=str, help="Download only a specific model (primary, embedding, classifier)")

    # Add direct URL download support
    parser.add_argument("--url", type=str, help="Direct URL to download")
    parser.add_argument("--output", type=str, help="Output filename for direct URL download (used with --url)")

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger("model-downloader").setLevel(logging.DEBUG)

    # Apply system optimizations
    if not args.skip_optimizations:
        await optimize_system()

    # Create download manager with common settings
    downloader = DownloadManager(
        output_dir=args.output_dir,
        parallel_downloads=args.parallel_downloads,
        parallel_chunks=args.parallel_chunks,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        force_redownload=args.force,
        azure_region=args.azure_region,
        tcp_optimization=not args.skip_optimizations,
    )

    # Handle direct URL download if specified
    if args.url:
        # Get output filename
        if args.output:
            output_file = args.output
        else:
            # Extract filename from URL
            parsed_url = urlparse(args.url)
            output_file = os.path.basename(unquote(parsed_url.path))
            if not output_file:
                output_file = "downloaded_file"  # Fallback name

        # Create output path
        output_path = Path(args.output_dir) / output_file

        logger.info(f"Starting direct URL download: {args.url} -> {output_path}")

        # Download the file
        success = await downloader.download_from_url(args.url, output_path)

        if success:
            logger.info(f"Successfully downloaded: {output_path}")
            sys.exit(0)
        else:
            logger.error(f"Failed to download: {args.url}")
            sys.exit(1)

    # Check if config file exists for regular downloads
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Optional: filter for specific model
    if args.model:
        logger.info(f"Only downloading model type: {args.model}")

        # Process config file
        all_tasks = await downloader.process_model_config(args.config)

        # Filter tasks for specified model
        filtered_tasks = [task for task in all_tasks if task["model_type"] == args.model]

        if not filtered_tasks:
            logger.error(f"No model found with type '{args.model}' in config")
            sys.exit(1)

        # Download only filtered tasks
        results = []
        for task in filtered_tasks:
            result = await downloader.download_task(task)
            results.append(result)

        success = all(results)
    else:
        # Run the downloader for all models
        success = await downloader.run(args.config)

    if success:
        logger.info("All downloads completed successfully")
        sys.exit(0)
    else:
        logger.error("Some downloads failed")
        sys.exit(1)


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        logger.error("This script requires Python 3.7 or higher")
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        sys.exit(1)
