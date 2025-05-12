#!/bin/bash

# Notebook Parameter Update Script
# This script applies notebook parameters to source code files
# Run from Git_soduku directory

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Error handling
handle_error() {
    echo -e "${RED}ERROR: $1${NC}"
    exit 1
}

# Check directory
if [ ! -d "models" ] || [ ! -d "config" ]; then
    handle_error "Please run from Git_soduku directory"
fi

# Create backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="backups_${TIMESTAMP}"
mkdir -p "$BACKUP_DIR" || handle_error "Failed to create backup directory"
echo -e "${BLUE}Created backup: $BACKUP_DIR${NC}"

# Backup files
cp models/intersection_detector.py "$BACKUP_DIR/" || handle_error "Backup failed"
cp models/cell_extractor.py "$BACKUP_DIR/" || handle_error "Backup failed"
cp config/default_fallbacks.py "$BACKUP_DIR/" || handle_error "Backup failed"
echo -e "${GREEN}Files backed up successfully${NC}"

# Update confidence threshold
sed -i 's/self\.confidence_threshold = self\.settings\.get("confidence_threshold", 0\.7)/self.confidence_threshold = self.settings.get("confidence_threshold", 0.77)/' models/intersection_detector.py

# Simplify preprocessing
sed -i '/_preprocess_image/,/return normalized/ c\
    def _preprocess_image(self, image: ImageType) -> ImageType:\
        """\
        Preprocess image for intersection detection.\
        Modified to match notebook approach.\
        """\
        # SIMPLIFIED to match notebook\
        normalized = image.astype(np.float32) / 255.0\
        return normalized' models/intersection_detector.py

# Update border padding
sed -i 's/self\.border_padding = self\.settings\.get("border_padding", 0\.07)/self.border_padding = self.settings.get("border_padding", 0.05)/' models/cell_extractor.py

# Disable multiple extractors
sed -i 's/self\.use_multiple_extractors = self\.settings\.get("use_multiple_extractors", True)/self.use_multiple_extractors = self.settings.get("use_multiple_extractors", False)/' models/cell_extractor.py

# Force extract settings (simplified to avoid shell issues)
cat > /tmp/extract_method.txt << 'EOF'
    @robust_method(max_retries=3, timeout_sec=60.0)
    def extract(self, image: ImageType, grid_points: GridPointsType) -> List[List[ImageType]]:
        """
        Extract cells using forced perspective extractor settings.
        """
        try:
            validate_image(image)
            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise CellExtractionError(f"Invalid grid points: {len(grid_points)} rows, expected 10")

            # Force settings to match notebook
            self.perspective_extractor.extraction_mode = "preserve"
            self.perspective_extractor.adaptive_thresholding = False
            self.perspective_extractor.contrast_enhancement = False
            self.perspective_extractor.noise_reduction = False
            self.perspective_extractor.border_padding = 0.05
            self.use_multiple_extractors = False
            
            # Use only perspective extractor
            return self.perspective_extractor.extract(image, grid_points)
            
        except Exception as e:
            if isinstance(e, CellExtractionError): raise
            raise CellExtractionError(f"Error in robust cell extraction: {str(e)}")
EOF

sed -i '/def extract(self, image: ImageType, grid_points: GridPointsType)/,/return cell_images/ {//!d}' models/cell_extractor.py
sed -i '/def extract(self, image: ImageType, grid_points: GridPointsType)/ {r /tmp/extract_method.txt
d}' models/cell_extractor.py

# Update config
sed -i 's/"confidence_threshold": 0\.7,/"confidence_threshold": 0.77,/' config/default_fallbacks.py
sed -i 's/"min_intersections": 650,/"min_intersections": 60,/' config/default_fallbacks.py
sed -i 's/"border_padding": 0\.1,/"border_padding": 0.05,/' config/default_fallbacks.py
sed -i 's/"contrast_enhancement": True,/"contrast_enhancement": False,/' config/default_fallbacks.py
sed -i 's/"noise_reduction": True,/"noise_reduction": False,/' config/default_fallbacks.py
sed -i 's/"adaptive_thresholding": True,/"adaptive_thresholding": False,/' config/default_fallbacks.py
sed -i 's/"histogram_equalization": True,/"histogram_equalization": False,/' config/default_fallbacks.py
sed -i 's/"use_multiple_extractors": True,/"use_multiple_extractors": False,/' config/default_fallbacks.py

echo -e "${GREEN}All changes applied successfully!${NC}"
echo -e "${BLUE}Original files backed up to: $BACKUP_DIR${NC}"
