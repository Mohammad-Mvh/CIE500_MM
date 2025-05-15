from qgis.core import QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer, QgsProcessingParameterRasterDestination
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
import os

# Ensure the output directory exists before saving the raster
output_folder = os.path.dirname(output_raster)


class SimpleFloodMap(QgsProcessingAlgorithm):
    DEM_INPUT = "DEM_INPUT"
    OUTPUT_RASTER = "OUTPUT_RASTER"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    def initAlgorithm(self, config=None):
        """Define input parameters"""
        self.addParameter(QgsProcessingParameterRasterLayer(self.DEM_INPUT, "Input DEM"))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT_RASTER, "Output Flood Depth Raster"))

    def processAlgorithm(self, parameters, context, feedback):
        """Main execution"""
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM_INPUT, context)
        output_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)

        # Ensure DEM is valid
        if not dem_layer:
            feedback.reportError("❌ DEM layer is invalid!")
            return {}

        # Define raster calculation formula (Fixed Water Level = 35m)
        formula = "(35 - A) * (35 > A)"

        # Create raster calculator entry
        entry = QgsRasterCalculatorEntry()
        entry.ref = "A"
        entry.raster = dem_layer
        entry.bandNumber = 1

        # Run the Raster Calculator
        calculator = QgsRasterCalculator(
            formula, output_raster, "GTiff",  # Explicitly setting "GTiff" format
            dem_layer.extent(), dem_layer.width(), dem_layer.height(), [entry]
        )

        result = calculator.processCalculation()
        if result != 0:
            feedback.reportError("❌ Error during raster calculation!")
            return {}

        feedback.pushInfo(f"✅ Raster successfully created: {output_raster}")
        return {self.OUTPUT_RASTER: output_raster}

    def name(self):
        return "simple_flood_map"

    def displayName(self):
        return "Simple Flood Depth Map"

    def group(self):
        return "Custom Scripts"

    def groupId(self):
        return "custom_scripts"

    def createInstance(self):
        return SimpleFloodMap()
