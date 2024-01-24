import nbconvert.exporters
import nbconvert.preprocessors
import nbconvert.writers

TAG_CONVERT_MODULE = "convert-module"
NAME_MODULE = "transformer"


class ConvertModulePreprocessor(nbconvert.preprocessors.Preprocessor):
    def preprocess(self, nb, resources):
        '''Select only cells with correct tag.'''
        nb.cells = [
            cell for cell in nb.cells
            if TAG_CONVERT_MODULE in cell["metadata"].get("tags", [])
        ]
        return nb, resources


if __name__ == "__main__":
    exporter = nbconvert.exporters.PythonExporter()
    exporter.exclude_input_prompt = True
    exporter.register_preprocessor(ConvertModulePreprocessor(), True)
    (output, resources) = exporter.from_filename("project.ipynb")
    writer = nbconvert.writers.FilesWriter()
    writer.write(output, resources, NAME_MODULE)
