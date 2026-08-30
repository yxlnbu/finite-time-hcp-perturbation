"""Micromorphic/GND reference implementation for the autonomous HCP CP project."""

from .micromorphic import (
    Hex8MicromorphicElement,
    MicromorphicParameters,
    MicromorphicResponse,
    cross_matrix,
    evaluate_micromorphic_state,
)
from .grain_boundary import GrainBoundaryResponse, evaluate_grain_boundary_state
from .monolithic_element import (
    CondensedPointResponse,
    Hex8MonolithicAssembler,
    MonolithicElementAssembly,
    MonolithicElementParameters,
    PointFieldState,
)
from .state_contract import (
    LocalState92,
    STATE_SCHEMA,
    TENSOR_COMPONENT_ORDER,
    initial_local_state,
)
from .coupled_point_adapter import (
    GATE01_POINT_SOURCE_SCHEMA,
    CondensedMicromorphicPointAdapter,
    Gate01PointSourceProbe,
)
from .sl3_chart import N_SL3, SL3LocalChart, traceless_generators
from .branch_audit import (
    SpectralAdmissibilityError,
    SpectralBranchAudit,
    SpectralNondifferentiableError,
)
from .spectral_export import (
    INPUT_COORDINATE_LABELS,
    N_ACTIVE,
    N_FIELD,
    N_SPECTRAL,
    OUTPUT_COORDINATE_LABELS,
    SPECTRAL_CHART_SCHEMA,
    SPECTRAL_EXPORT_SCHEMA,
    SPECTRAL_STATE_SCHEMA,
    CoordinateBlock,
    ContinuousSpectralPointModel,
    SpectralActiveState62,
    SpectralDerivativeOptions,
    SpectralObserverState,
    SpectralPointDerivatives,
    SpectralPointExport,
    spectral_point_export,
)
from .qs_descriptor import (
    N_QS,
    QS_DESCRIPTOR_SCHEMA,
    QSDescriptorAdmission,
    QSDescriptorAssembly,
    QSDescriptorMetadata,
    assemble_qs_descriptor,
    direction_maps,
)
from .state_contract_v2 import (
    LocalState105,
    N_STATE105,
    STATE105_SCHEMA,
    initial_local_state105,
    upgrade_state92_to_state105,
)
from .twin_damage_drx_v1 import (
    EXTENDED_ACTIVE_SCHEMA,
    MechanismInputsV1,
    MechanismParametersV1,
    MechanismPointStateV1,
    advance_backward_euler,
    mechanism_response,
)

__all__ = [
    "Hex8MicromorphicElement",
    "MicromorphicParameters",
    "MicromorphicResponse",
    "cross_matrix",
    "evaluate_micromorphic_state",
    "GrainBoundaryResponse",
    "evaluate_grain_boundary_state",
    "CondensedPointResponse",
    "Hex8MonolithicAssembler",
    "MonolithicElementAssembly",
    "MonolithicElementParameters",
    "PointFieldState",
    "LocalState92",
    "STATE_SCHEMA",
    "TENSOR_COMPONENT_ORDER",
    "initial_local_state",
    "CondensedMicromorphicPointAdapter",
    "N_SL3",
    "SL3LocalChart",
    "traceless_generators",
    "SpectralAdmissibilityError",
    "SpectralBranchAudit",
    "SpectralNondifferentiableError",
    "INPUT_COORDINATE_LABELS",
    "N_ACTIVE",
    "N_FIELD",
    "N_SPECTRAL",
    "OUTPUT_COORDINATE_LABELS",
    "SPECTRAL_CHART_SCHEMA",
    "SPECTRAL_EXPORT_SCHEMA",
    "SPECTRAL_STATE_SCHEMA",
    "CoordinateBlock",
    "ContinuousSpectralPointModel",
    "SpectralActiveState62",
    "SpectralDerivativeOptions",
    "SpectralObserverState",
    "SpectralPointDerivatives",
    "SpectralPointExport",
    "spectral_point_export",
    "N_QS",
    "QS_DESCRIPTOR_SCHEMA",
    "QSDescriptorAdmission",
    "QSDescriptorAssembly",
    "QSDescriptorMetadata",
    "assemble_qs_descriptor",
    "direction_maps",
    "LocalState105",
    "N_STATE105",
    "STATE105_SCHEMA",
    "initial_local_state105",
    "upgrade_state92_to_state105",
    "EXTENDED_ACTIVE_SCHEMA",
    "MechanismInputsV1",
    "MechanismParametersV1",
    "MechanismPointStateV1",
    "advance_backward_euler",
    "mechanism_response",
]

__version__ = "0.2.0.dev0"
