from qgis.core import QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer, QgsProcessingParameterFolderDestination
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
from qgis.core import QgsRasterBandStats
import os

class FloodInundationModel(QgsProcessingAlgorithm):
    DEM_INPUT = "DEM_INPUT"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    def initAlgorithm(self, config=None):
        """Define input parameters"""
        self.addParameter(QgsProcessingParameterRasterLayer(self.DEM_INPUT, "Input DEM"))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, "Output Folder"))

    def processAlgorithm(self, parameters, context, feedback):
        """Main execution method"""
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM_INPUT, context)
        output_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)

        # Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Check if DEM layer is valid
        if not dem_layer:
            feedback.reportError("❌ DEM layer is invalid. Please check the input.")
            return {}
        provider = dem_layer.dataProvider()
        stats = provider.bandStatistics(1, QgsRasterBandStats.All)
        dem_min = stats.minimumValue
        dem_max = stats.maximumValue

        feedback.pushInfo(f"DEM Min: {dem_min}, DEM Max: {dem_max}")

        if any(water_level < dem_min for _, water_level in water_levels):
            feedback.reportError("❌ Some water levels are below the DEM min elevation!")
            return {}

        # Check DEM properties
        feedback.pushInfo(f"📏 DEM Extent: {dem_layer.extent().toString()}")
        feedback.pushInfo(f"🗺️ DEM CRS: {dem_layer.crs().authid()}")
        feedback.pushInfo(f"🟢 DEM Width x Height: {dem_layer.width()} x {dem_layer.height()}")

        if dem_layer.width() == 0 or dem_layer.height() == 0:
            feedback.reportError("❌ DEM has zero width or height! Check the dataset.")
            return {}

        # Define time series of water elevations
        water_levels = [
            (0, 20.0), (2, 30.0), (4, 40.0), (6, 50.0), (8, 55.0), (10, 60.0),
            (12, 65.0), (14, 55.0), (16, 45.0), (18, 55.0), (20, 70.0), (22, 55.0),
            (24, 40.0), (26, 35.0), (28, 20.0),
        ]
        if not dem_layer.isValid():
            feedback.reportError("❌ DEM layer is not valid. Check the dataset.")
            return {}
        if not os.access(output_folder, os.W_OK):
            feedback.reportError(f"❌ Cannot write to output folder: {output_folder}")
            return {}
        feedback.pushInfo(f"DEM Band Count: {dem_layer.bandCount()}")
        feedback.pushInfo(f"DEM NoData Value: {dem_layer.dataProvider().sourceNoDataValue(1)}")
        for time, water_level in water_levels:
            output_path = os.path.join(output_folder, f"flood_depth_{time}hr.tif")

            # Skip existing files
            if os.path.exists(output_path):
                feedback.pushInfo(f"Skipping {output_path}, file already exists.")
                continue

            # Define Raster Calculator formula
            formula = f"(({water_level} - A) * (A < {water_level}))"

            # Set up Raster Calculator entry
            entry = QgsRasterCalculatorEntry()
            entry.ref = "A"
            entry.raster = dem_layer
            entry.bandNumber = 1

            # Run Raster Calculator
            calculator = QgsRasterCalculator(
                formula, output_path, "GTiff",
                dem_layer.extent(), dem_layer.width(), dem_layer.height(), [entry]
            )

            result = calculator.processCalculation()

            # Check result and report any errors
            if result == 0:
                feedback.pushInfo(f"✅ Successfully created: {output_path}")
            else:
                feedback.reportError(f"❌ Error processing {output_path}. Error code: {result}")

        return {}

    def name(self):
        return "flood_inundation_model"

    def displayName(self):
        return "Flood Inundation Model"

    def group(self):
        return "Custom Scripts"

    def groupId(self):
        return "custom_scripts"

    def createInstance(self):
        return FloodInundationModel()
