"""Generic exceptions for the graphragX pipeline foundation."""
from pipeline.exceptions import PipelineException


class UnsupportedDatasetSelectionException(PipelineException):
    """Raised when a preparation run requests an unsupported dataset."""


class InvalidMainLlmSelectionException(PipelineException):
    """Raised when an invalid main LLM option is provided."""


class InvalidLlmProviderSelectionException(PipelineException):
    """Raised when an invalid LLM provider is provided."""


class InvalidSubgraphConstructionSelectionException(PipelineException):
    """Raised when an invalid subgraph construction option is provided."""


class InvalidContextConstructionSelectionException(PipelineException):
    """Raised when an invalid context construction option is provided."""


class InvalidGnnLayerCountSelectionException(PipelineException):
    """Raised when an invalid GNN layer count option is provided."""


class InvalidGnnArchitectureSelectionException(PipelineException):
    """Raised when an invalid GNN architecture is selected."""


class InvalidGnnArchitectureConfigurationException(PipelineException):
    """Raised when options are incompatible with the selected architecture."""


class InvalidGnnHiddenDimensionSelectionException(PipelineException):
    """Raised when an invalid GNN hidden dimension option is provided."""


class InvalidNodeClassifierSelectionException(PipelineException):
    """Raised when an invalid node classifier option is provided."""


class InvalidQuestionEmbeddingModelSelectionException(PipelineException):
    """Raised when an invalid question embedding model option is provided."""


class InvalidRelationEmbeddingModelSelectionException(PipelineException):
    """Raised when an invalid relation embedding model option is provided."""


class InvalidEntityEmbeddingModelSelectionException(PipelineException):
    """Raised when an invalid entity embedding model option is provided."""


class InvalidInteractiveConfigurationInputException(PipelineException):
    """Raised when interactive configuration input cannot be read safely."""


class MissingHuggingFaceDatasetsDependencyException(PipelineException):
    """Raised when Hugging Face datasets is required but not installed."""


class UnsupportedDatasetLoaderException(PipelineException):
    """Raised when no supported dataset loader configuration exists."""


class DatasetLoadingException(PipelineException):
    """Raised when dataset loading fails."""


class MalformedDatasetException(PipelineException):
    """Raised when a loaded dataset does not expose the expected structure."""


class UnsupportedDatasetProcessorException(PipelineException):
    """Raised when no supported dataset processor exists for a dataset."""


class MalformedWebQSPExampleException(PipelineException):
    """Raised when a WebQSP example does not expose the expected structure."""


class ProcessedDatasetStorageException(PipelineException):
    """Raised when processed dataset cache storage fails."""


class OpenAiEmbeddingConfigurationException(PipelineException):
    """Raised when OpenAI embedding configuration is incomplete."""


class QdrantEmbeddingStoreException(PipelineException):
    """Raised when the Qdrant embedding store is unavailable or inconsistent."""


class GnnAnswerRetrieverTrainingException(PipelineException):
    """Raised when GNN answer-retriever training fails."""


class GnnAnswerRetrieverModelRunException(PipelineException):
    """Raised when a saved GNN answer-retriever run cannot be resolved or loaded."""


class GnnAnswerRetrieverEvaluationException(PipelineException):
    """Raised when GNN answer-retriever evaluation fails."""
