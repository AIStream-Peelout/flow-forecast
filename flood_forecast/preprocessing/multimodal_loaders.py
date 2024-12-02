from torch.utils.data import Dataset
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import os
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
from flood_forecast.preprocessing.pytorch_loaders import CSVDataLoader, CSVTestLoader


class ImageSequenceHelper:
    """Helper class to handle loading and processing of image sequences"""

    def __init__(
            self,
            cache_dir: Optional[str] = None,
            transform: Optional[transforms.Compose] = None,
            cloud_config: Optional[Dict] = None
    ):
        """Initialize the image sequence helper

        Args:
            cache_dir: Directory to cache downloaded files
            transform: Torchvision transforms to apply to images
            cloud_config: Configuration for cloud storage access
        """
        self.cache_dir = cache_dir if cache_dir else "./.cache"
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.cloud_config = cloud_config
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_from_cloud(self, path: str) -> str:
        """Download file from cloud storage if needed"""
        if path.startswith(("gs://", "s3://", "azure://")):
            local_path = os.path.join(self.cache_dir, Path(path).name)
            if not os.path.exists(local_path):
                # Implement cloud download logic based on path prefix
                if path.startswith("gs://"):
                    # Google Cloud Storage logic
                    pass
                elif path.startswith("s3://"):
                    # AWS S3 logic
                    pass
                elif path.startswith("azure://"):
                    # Azure Blob Storage logic
                    pass
            return local_path
        return path

    def load_context_sequence(
            self,
            image_paths: List[str],
            fill_method: str = "zero"
    ) -> torch.Tensor:
        """Load a sequence of images and handle missing values

        Args:
            image_paths: List of paths to images
            fill_method: Method to handle missing images ('zero', 'previous', 'interpolate')

        Returns:
            Tensor of shape (sequence_length, channels, height, width)
        """
        images = []
        previous_img = None

        for path in image_paths:
            if not path or pd.isna(path):
                if fill_method == "zero":
                    img_tensor = torch.zeros((3, 224, 224))
                elif fill_method == "previous" and previous_img is not None:
                    img_tensor = previous_img
                else:
                    img_tensor = torch.zeros((3, 224, 224))
            else:
                local_path = self.download_from_cloud(path)
                img = Image.open(local_path).convert('RGB')
                img_tensor = self.transform(img)
                previous_img = img_tensor

            images.append(img_tensor)

        return torch.stack(images)


class UniformMultiModalLoader(CSVDataLoader):
    """A data loader for multimodal time series data where different modalities
    are aligned with the time series at regular intervals."""

    def __init__(
            self,
            main_params: Dict[str, Any],
            image_config: Optional[Dict[str, Any]] = None,
            text_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the UniformMultiModal data loader

        Args:
            main_params: Parameters for the base CSVDataLoader
            image_config: Configuration for image modality, should include:
                - path_col: Column name containing image paths
                - cache_dir: Directory to cache images
                - fill_method: Method to handle missing images
                - transform: Optional torchvision transforms
            text_config: Configuration for text modality (future extension)
        """
        super().__init__(**main_params)

        # Initialize image helper if image config is provided
        self.image_helper = None
        self.image_path_col = None
        if image_config:
            self.image_helper = ImageSequenceHelper(
                cache_dir=image_config.get('cache_dir'),
                transform=image_config.get('transform'),
                cloud_config=image_config.get('cloud_config')
            )
            self.image_path_col = image_config['path_col']

            # Validate that image path column exists
            if self.image_path_col not in self.df.columns:
                raise ValueError(f"Image path column {self.image_path_col} not found in dataframe")

        # Store configs for potential future use
        self.image_config = image_config
        self.text_config = text_config

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single item from the dataset

        Args:
            idx: Index to retrieve

        Returns:
            Tuple of:
                - Dictionary containing:
                    - 'time_series': Time series tensor from base CSVDataLoader
                    - 'images': Image sequence tensor if image_config provided
                    - (future) 'text': Text tensor if text_config provided
                - Target tensor
        """
        # Get base time series data from parent class
        ts_src_data, ts_trg_dat = super().__getitem__(idx)

        # Initialize return dictionary with time series data
        return_dict = {
            'time_series': ts_src_data
        }

        # Add image data if configured
        if self.image_helper:
            image_paths = self.df[self.image_path_col].iloc[
                          idx: self.forecast_history + idx
                          ].tolist()

            images = self.image_helper.load_context_sequence(
                image_paths,
                fill_method=self.image_config.get('fill_method', 'zero')
            )
            return_dict['images'] = images

        # Text modality could be added here in the future

        return return_dict, ts_trg_dat

    def get_from_start_date(self, forecast_start: datetime):
        """Get data starting from a specific date

        This overrides the parent method to ensure proper handling of all modalities
        """
        dt_row = self.original_df[
            self.original_df["datetime"] == forecast_start
        ]
        revised_index = dt_row.index[0]
        return self.__getitem__(revised_index - self.forecast_history)


class UniformMultiModalTestLoader(CSVTestLoader):
    """Test data loader for multimodal time series data"""

    def __init__(
            self,
            df_path: str,
            forecast_total: int,
            image_config: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        """Initialize the test loader

        Args:
            df_path: Path to the CSV file
            forecast_total: Total forecast length
            image_config: Configuration for image modality
            **kwargs: Additional arguments for CSVTestLoader
        """
        super().__init__(df_path, forecast_total, **kwargs)

        # Initialize image helper if needed
        self.image_helper = None
        self.image_path_col = None
        if image_config:
            self.image_helper = ImageSequenceHelper(
                cache_dir=image_config.get('cache_dir'),
                transform=image_config.get('transform'),
                cloud_config=image_config.get('cloud_config')
            )
            self.image_path_col = image_config['path_col']

    def __getitem__(self, idx):
        """Get a test item including all modalities"""
        # Get base time series data
        if self.target_supplied:
            historical_rows = self.df.iloc[idx: self.forecast_history + idx]
            target_idx_start = self.forecast_history + idx

            all_rows_orig = self.original_df.iloc[
                            idx: self.forecast_total + target_idx_start
                            ].copy()

            return_dict = {
                'time_series': torch.from_numpy(historical_rows.to_numpy()).float()
            }

            # Add image data if configured
            if self.image_helper:
                image_paths = self.df[self.image_path_col].iloc[
                              idx: self.forecast_history + idx
                              ].tolist()

                images = self.image_helper.load_context_sequence(
                    image_paths,
                    fill_method='zero'
                )
                return_dict['images'] = images

            return return_dict, all_rows_orig, target_idx_start
