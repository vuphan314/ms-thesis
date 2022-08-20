#!/usr/bin/env python3

from definitions import *

################################################################################
'''Defines general switches.'''

WRITING_DATA_TO_PY_FILE = 0

INCLUDING_LINEAR = 0

SHOWING_WARNINGS = 0

################################################################################
'''Defines switches for analysis.'''

INCLUDING_OVERFLOW_COMPLETIONS = 0

FLOAT_TOLERANCE_THRESHOLD = 1
FLOAT_DIFF_TOLERANCE = 1e-3 # BIG - SMALL <= DIFF if SMALL == 0. or BIG <= THRESHOLD
FLOAT_RATIO_TOLERANCE = 1 + 1e-3 # BIG / SMALL <= RATIO otherwise

################################################################################
'''Defines global variables.'''

globalCachetDummyModelCounts1 = []
globalCachetDummyModelCounts2 = []

## getOutFileReports sets these once then gets them afterward (caching):
### exp1:
globalReportsExp1AddmcBayesMinic2d = [] # globalReports<Exp><Counter><BenchmarkFam><WeightFormat>
globalReportsExp1AddmcPseudoweightedMinic2d = []
### exp1b:
globalReportsExp1BAddmcBayesMinic2d = []
globalReportsExp1BAddmcPseudoweightedMinic2d = []
### exp2 BAYES:
globalReportsExp2AddmcBayesMinic2d = []
globalReportsExp2C2dBayesDdnnf = []
globalReportsExp2CachetBayesCachet = []
globalReportsExp2D4BayesDdnnf = []
globalReportsExp2Minic2dBayesMinic2d = []
### exp2 PSEUDOWEIGHTED:
globalReportsExp2AddmcPseudoweightedMinic2d = []
globalReportsExp2C2dPseudoweightedDdnnf = []
globalReportsExp2CachetPseudoweightedCachet = []
globalReportsExp2D4PseudoweightedDdnnf = []
globalReportsExp2Minic2dPseudoweightedMinic2d = []

################################################################################
'''Defines classes.'''

DUMMY_MODEL_COUNT = -1.

@enum.unique
class ClusteringHeuristic(enum.Enum):
    MONOLITHIC = '1'
    LINEAR = '2'
    BUCKET_LIST = '3'
    BUCKET_TREE = '4'
    BOUQUET_LIST = '5'
    BOUQUET_TREE = '6'

@enum.unique
class VarOrderingHeuristic(enum.Enum):
    # APPEARANCE = '1'
    # DECLARATION = '2'
    RANDOM = '3'
    MCS = '4'
    LEXP = '5'
    LEXM = '6'

@enum.unique
class Counter(enum.Enum):
    ADDMC = 'ADDMC'
    C2D = 'c2d'
    CACHET = 'Cachet'
    D4 = 'd4'
    MINIC2D = 'miniC2D'
    SHARPSAT = 'sharpSAT'
    DUMMY = 'DUMMY_COUNTER'

    def __str__(self):
        return self.value

    @classmethod
    def getWeightedCounters(cls): # list
        return [
            cls.ADDMC,
            cls.C2D,
            cls.CACHET,
            cls.D4,
            cls.MINIC2D,
        ]

@enum.unique
class WeightFormat(enum.Enum):
    # UNWEIGHTED = '1'
    MINIC2D = '2'
    CACHET = '3'
    DDNNF = 'DDNNF_WEIGHT_FORMAT' # for c2d (with d-dnnf-reasoner) and d4
    WEIGHTED = 'WEIGHTED_WEIGHT_FORMAT'
    DUMMY = 'DUMMY_WEIGHT_FORMAT'

    def isWeighted(self):
        return self in [
            WeightFormat.MINIC2D,
            WeightFormat.CACHET,
            WeightFormat.DDNNF,
            WeightFormat.WEIGHTED,
        ]

    @classmethod
    def getSpecificWeightedFormat(cls, counter):
        if counter == Counter.ADDMC:
            return cls.MINIC2D
        elif counter == Counter.C2D:
            return cls.DDNNF
        elif counter == Counter.CACHET:
            return cls.CACHET
        elif counter == Counter.D4:
            return cls.DDNNF
        elif counter == Counter.MINIC2D:
            return cls.MINIC2D
        elif counter == Counter.SHARPSAT:
            return cls.DUMMY
        else:
            raiseWrongCounterException(counter)

@enum.unique
class BenchmarkFam(enum.Enum):
    BAYES = 'Bayes'
    PSEUDOWEIGHTED = 'pseudoweighted'
    ALTOGETHER = 'altogether'

@enum.unique
class Experiment(enum.Enum):
    ALL_CONFIGS = '1a'
    BEST_CONFIGS = '1b'
    COUNTERS = '2'

    def __repr__(self):
        return '{}.{}'.format(self.__class__.__name__, self.name)

    def withConfigs(self):
        return self in {Experiment.ALL_CONFIGS, Experiment.BEST_CONFIGS}

@enum.unique
class OutFileStatus(enum.Enum):
    COMPLETION = 'solved'
    TIMEOUT = 'timedOut'
    SPACEOUT = 'memedOut' # warns
    KILL = 'killed' # warns
    ERROR = 'err' # warns
    DUMMY = 'DUMMY_OUT_FILE_STATUS'

    def __str__(self):
        return self.value

    def __repr__(self):
        return '{}.{}'.format(self.__class__.__name__, self.name)

class OutFileReport:
    def __init__(self, outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, counter, experiment):
        assert outFileStatus in OutFileStatus
        self.outFilePath = outFilePath
        self.outFileStatus = outFileStatus
        self.cnfFileName = cnfFileName
        self.modelCount = modelCount
        self.overflow = overflow
        self.time = time
        self.counter = counter
        self.experiment = experiment

    def __str__(self):
        return '({})'.format(', '.join([str(field) for field in [
            self.outFilePath,
            self.outFileStatus,
            self.cnfFileName,
            self.modelCount,
            self.overflow,
            self.time,
            self.counter,
        ]]))

    def isComplete(self):
        complete = self.outFileStatus == OutFileStatus.COMPLETION
        complete &= self.time <= getTimeLimit(self.experiment)
        if not INCLUDING_OVERFLOW_COMPLETIONS:
            complete &= not self.overflow
        return complete

    @classmethod
    def getDummy(cls):
        return cls('DUMMY_OUT_FILE_PATH', OutFileStatus.DUMMY, 'DUMMY_CNF_FILE_NAME', DUMMY_MODEL_COUNT, True, INF, Counter.DUMMY)

    @classmethod
    def reprFloat(cls, f):
        return 'INF' if math.isinf(f) else repr(f)

    @classmethod
    def reprField(cls, field):
        return cls.reprFloat(field) if type(field) == float else repr(field)

class AddmcOutFileReport(OutFileReport):
    def __init__(self, outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, heuristicTuple, mavc, experiment):
        super().__init__(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, Counter.ADDMC, experiment)
        self.heuristicTuple = heuristicTuple
        self.mavc = mavc
        self.experiment = experiment

    def __str__(self):
        return '({})'.format(', '.join([str(field) for field in [
            # self.outFilePath,
            # self.outFileStatus,
            self.cnfFileName,
            self.modelCount,
            # self.overflow,
            self.time,
            getNamedHeuristicTuple(self.heuristicTuple),
            # self.mavc,
        ]]))

    def __repr__(self):
        fields = [super(self.__class__, self).reprField(field) for field in [
            self.outFilePath,
            self.outFileStatus,
            self.cnfFileName,
            self.modelCount,
            self.overflow,
            self.time,
            self.heuristicTuple,
            self.mavc,
            self.experiment,
        ]]
        return '{}({})'.format(self.__class__.__name__, ', '.join(fields))

class C2dOutFileReport(OutFileReport):
    def __init__(self, outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, compilerTime):
        super().__init__(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, Counter.C2D, Experiment.COUNTERS)
        self.compilerTime = compilerTime

class CachetOutFileReport(OutFileReport):
    def __init__(self, outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time):
        super().__init__(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, Counter.CACHET, Experiment.COUNTERS)

class D4OutFileReport(OutFileReport):
    def __init__(self, outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time):
        super().__init__(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, Counter.D4, Experiment.COUNTERS)

class Minic2dOutFileReport(OutFileReport):
    def __init__(self, outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time):
        super().__init__(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, Counter.MINIC2D, Experiment.COUNTERS)

class SharpsatOutFileReport(OutFileReport):
    def __init__(self, outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time):
        super().__init__(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, Counter.SHARPSAT, Experiment.COUNTERS)

################################################################################
'''Helps.'''

def isPseudoweightedBenchmark(cnfFileName):
    assert type(cnfFileName) == str
    return cnfFileName in PSEUDOWEIGHTED_VAR_COUNTS

def isBayesBenchmark(cnfFileName):
    return not isPseudoweightedBenchmark(cnfFileName)

def getBenchmarkCount(benchmarkFam): # 1976 benchmarks
    if benchmarkFam == BenchmarkFam.BAYES:
        return 1080 # 660 DQMR, 420 Grid
    elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
        return 896 # 36 BayesianNetwork, 18 bmc, 65 circuit, 35 Configuration, 68 Handmade, 557 Planning, 7 qif, 104 random, 6 scheduling)
    elif benchmarkFam == BenchmarkFam.ALTOGETHER:
        benchmarkCount = getBenchmarkCount(BenchmarkFam.BAYES) + getBenchmarkCount(BenchmarkFam.PSEUDOWEIGHTED)
        return benchmarkCount # 1976
    else:
        raiseWrongBenchmarkFamException(benchmarkFam)
def getBenchmarkCount(benchmarkFam): # 1914 benchmarks
    if benchmarkFam == BenchmarkFam.BAYES:
        return 1091 # 660 DQMR, 420 Grid, 11 Plan_Recognition
    elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
        return 823 # 546 PLANNING, 277 REST (18 bmc, 39 circuit, 35 Configuration, 68 Handmade, 7 qif, 104 random, 6 scheduling)
    elif benchmarkFam == BenchmarkFam.ALTOGETHER:
        benchmarkCount = getBenchmarkCount(BenchmarkFam.BAYES) + getBenchmarkCount(BenchmarkFam.PSEUDOWEIGHTED)
        return benchmarkCount # 1914
    else:
        raiseWrongBenchmarkFamException(benchmarkFam)

def getCompletionRateStr(benchmarkFam, completionCount, columnWidth): # percentage
    completionRate = completionCount / getBenchmarkCount(benchmarkFam) * 100
    return FLOAT_FORMAT.format(completionRate).rjust(columnWidth)

def getTimeLimit(experiment):
    if experiment == Experiment.ALL_CONFIGS:
        return 10
    elif experiment in {Experiment.BEST_CONFIGS, Experiment.COUNTERS}:
        return 1000
    else:
        raiseWrongExperimentException(experiment)

def getTimeLimitStr(experiment):
    return '{}-second'.format(getTimeLimit(experiment))

def getTime():
    return datetime.datetime.now()

def getDuration(startTime):
    return (getTime() - startTime).total_seconds()

def getFileName(filePath):
    return os.path.basename(filePath)

def getBaseFileName(filePath):
    return os.path.splitext(getFileName(filePath))[0]

def getDirPath(filePath):
    return os.path.dirname(filePath)

def getWeightFilePath(minic2dCnfFilePath):
    weightDirPath = getDirPath(minic2dCnfFilePath).replace('_MINIC2D', '_w')
    weightFileName = getBaseFileName(minic2dCnfFilePath) + '.w'
    return os.path.join(weightDirPath, weightFileName)

def getNnfFilePath(minic2dCnfFilePath): # will be written by C2D
    return minic2dCnfFilePath + '.nnf'

def getQueryFilePath(minic2dCnfFilePath): # writes query file for d-DNNF-reasoner
    queryFilePath = minic2dCnfFilePath.replace('.cnf', '.query')
    with open(queryFilePath, 'w') as queryFile:
        lines = [
            'load {}'.format(getNnfFilePath(minic2dCnfFilePath)),
            'w {}'.format(getWeightFilePath(minic2dCnfFilePath)),
            'mc'
        ]
        lines = '\n'.join(lines) + '\n\n'
        queryFile.write(lines)
    return queryFilePath

def getFilePaths(nestedDirPath, fileExtension='.out', givenFileNames=[], emptyOk=True):
    filePaths = []
    for (dirPath, _, fileNames) in os.walk(nestedDirPath):
        for fileName in fileNames:
            if givenFileNames:
                if fileName in givenFileNames:
                    filePaths.append(os.path.join(dirPath, fileName))
            else:
                if fileName.endswith('{}'.format(fileExtension)):
                    filePaths.append(os.path.join(dirPath, fileName))
    filePaths = sorted(filePaths)
    if not filePaths and not emptyOk:
        raise ExperimentException('{}: no file name matches "*{}"{}'.format(nestedDirPath, fileExtension, ' and is in {}'.format(givenFileNames) if givenFileNames else ''))
    return filePaths

def getCnfFilePaths(benchmarkFam, weightFormat, counter):
    if benchmarkFam == BenchmarkFam.BAYES:
        cnfDirPath = BAYES_BENCHMARKS_PATH
        if weightFormat != WeightFormat.CACHET:
            cnfDirPath += '_MINIC2D'
    elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
        cnfDirPath = PSEUDOWEIGHTED_BENCHMARKS_PATH
        if weightFormat != WeightFormat.CACHET:
            cnfDirPath += '_MINIC2D'
    elif benchmarkFam == BenchmarkFam.ALTOGETHER:
        return getCnfFilePaths(BenchmarkFam.BAYES, weightFormat, counter) + getCnfFilePaths(BenchmarkFam.PSEUDOWEIGHTED, weightFormat, counter)
    else:
        raiseWrongBenchmarkFamException(benchmarkFam)

    return getFilePaths(cnfDirPath, '.cnf', emptyOk=False)

def getOutFilePaths(experiment, counter, weightFormat, clusteringHeuristic=None, benchmarkFam=BenchmarkFam.BAYES):
    assert clusteringHeuristic == None or clusteringHeuristic in ClusteringHeuristic

    def getOutDirPath():
        if counter == Counter.ADDMC:
            if experiment == Experiment.ALL_CONFIGS:
                if benchmarkFam == BenchmarkFam.BAYES:
                    if weightFormat == WeightFormat.MINIC2D:
                        return OUT_DIR_PATH_EXP_1_ADDMC_BAYES_MINIC2D
                    else:
                        raiseWrongWeightFormatException(weightFormat)
                elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
                    if weightFormat == WeightFormat.MINIC2D:
                        return OUT_DIR_PATH_EXP_1_ADDMC_BAYES_PSEUDOWEIGHTED
                    else:
                        raiseWrongWeightFormatException(weightFormat)
                else:
                    raiseWrongBenchmarkFamException(benchmarkFam)
            elif experiment == Experiment.BEST_CONFIGS:
                if benchmarkFam == BenchmarkFam.BAYES:
                    if weightFormat == WeightFormat.MINIC2D:
                        return OUT_DIR_PATH_EXP_1_B_ADDMC_BAYES_MINIC2D
                    else:
                        raiseWrongWeightFormatException(weightFormat)
                elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
                    if weightFormat == WeightFormat.MINIC2D:
                        return OUT_DIR_PATH_EXP_1_B_ADDMC_BAYES_PSEUDOWEIGHTED
                    else:
                        raiseWrongWeightFormatException(weightFormat)
                else:
                    raiseWrongBenchmarkFamException(benchmarkFam)
            elif experiment == Experiment.COUNTERS:
                if benchmarkFam == BenchmarkFam.BAYES:
                    if weightFormat == WeightFormat.MINIC2D:
                        return OUT_DIR_PATH_EXP_2_ADDMC_BAYES_MINIC2D
                    else:
                        raiseWrongWeightFormatException(weightFormat)
                elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
                    if weightFormat == WeightFormat.MINIC2D:
                        return OUT_DIR_PATH_EXP_2_ADDMC_PSEUDOWEIGHTED_MINIC2D
                    else:
                        raise raiseWrongWeightFormatException(weightFormat)
                else:
                    raiseWrongBenchmarkFamException(benchmarkFam)
            else:
                raiseWrongExperimentException(experiment)
        elif counter == Counter.CACHET:
            if benchmarkFam == BenchmarkFam.BAYES:
                if weightFormat == WeightFormat.CACHET:
                    return OUT_DIR_PATH_EXP_2_CACHET_BAYES_CACHET
                else:
                    raiseWrongWeightFormatException(weightFormat)
            elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
                if weightFormat == WeightFormat.CACHET:
                    return OUT_DIR_PATH_EXP_2_CACHET_PSEUDOWEIGHTED_CACHET
                else:
                    raiseWrongWeightFormatException(weightFormat)
            else:
                raiseWrongBenchmarkFamException(benchmarkFam)
        elif counter == Counter.MINIC2D:
            if benchmarkFam == BenchmarkFam.BAYES:
                if weightFormat == WeightFormat.MINIC2D:
                    return OUT_DIR_PATH_EXP_2_MINIC2D_BAYES_MINIC2D
                else:
                    raise raiseWrongWeightFormatException(weightFormat)
            elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
                if weightFormat == WeightFormat.MINIC2D:
                    return OUT_DIR_PATH_EXP_2_MINIC2D_PSEUDOWEIGHTED_MINIC2D
                else:
                    raise raiseWrongWeightFormatException(weightFormat)
            else:
                raiseWrongBenchmarkFamException(benchmarkFam)
        elif counter == Counter.C2D:
            if benchmarkFam == BenchmarkFam.BAYES:
                if weightFormat == WeightFormat.DDNNF:
                    return OUT_DIR_PATH_EXP_2_C2D_BAYES_DDNNF
                else:
                    raise raiseWrongWeightFormatException(weightFormat)
            elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
                if weightFormat == WeightFormat.DDNNF:
                    return OUT_DIR_PATH_EXP_2_C2D_PSEUDOWEIGHTED_DDNNF
                else:
                    raise raiseWrongWeightFormatException(weightFormat)
            else:
                raiseWrongBenchmarkFamException(benchmarkFam)
        elif counter == Counter.D4:
            if benchmarkFam == BenchmarkFam.BAYES:
                if weightFormat == WeightFormat.DDNNF:
                    return OUT_DIR_PATH_EXP_2_D4_BAYES_DDNNF
                else:
                    raiseWrongWeightFormatException(weightFormat)
            elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
                if weightFormat == WeightFormat.DDNNF:
                    return OUT_DIR_PATH_EXP_2_D4_PSEUDOWEIGHTED_DDNNF
                else:
                    raise raiseWrongWeightFormatException(weightFormat)
            else:
                raiseWrongBenchmarkFamException(benchmarkFam)
        else:
            raiseWrongCounterException(counter)

    return getFilePaths(getOutDirPath(), '.out', emptyOk=False)

def getClusteringHeuristics():
    clusteringHeuristics = [
        ClusteringHeuristic.BOUQUET_TREE.value,
        ClusteringHeuristic.BOUQUET_LIST.value,
        ClusteringHeuristic.BUCKET_TREE.value,
        ClusteringHeuristic.BUCKET_LIST.value,
        ClusteringHeuristic.MONOLITHIC.value,
    ]
    if INCLUDING_LINEAR:
        clusteringHeuristics.append(ClusteringHeuristic.LINEAR.value)
    return clusteringHeuristics

def getVarOrderingHeuristics():
    directVarOrderingHeuristics = [
        varOrderingHeuristic.value for varOrderingHeuristic in VarOrderingHeuristic
    ]

    inverseVarOrderingHeuristics = [
        '-' + varOrderingHeuristic.value for varOrderingHeuristic in VarOrderingHeuristic
        if varOrderingHeuristic != VarOrderingHeuristic.RANDOM
    ]

    return directVarOrderingHeuristics + inverseVarOrderingHeuristics

def getClusteringHeuristic(namedHeuristicTuple):
    return ClusteringHeuristic[namedHeuristicTuple[0]]

def getHeuristicTuple(namedHeuristicTuple):
    if type(namedHeuristicTuple) == str:
        namedHeuristicTuple = tuple(namedHeuristicTuple.split())

    assert type(namedHeuristicTuple) == tuple and len(namedHeuristicTuple) == 5

    def getVarOrderingHeuristic(namedVarOrderingHeuristic, inverseVarOrdering):
        varOrderingHeuristic = VarOrderingHeuristic[namedVarOrderingHeuristic].value
        if int(inverseVarOrdering):
            varOrderingHeuristic = '-' + varOrderingHeuristic
        return varOrderingHeuristic

    (clusteringHeuristic, formulaVarOrderingHeuristic, inverseFormulaVarOrdering, addVarOrderingHeuristic, inverseAddVarOrdering) = namedHeuristicTuple

    heuristicTuple = (
        ClusteringHeuristic[clusteringHeuristic].value,
        getVarOrderingHeuristic(formulaVarOrderingHeuristic, inverseFormulaVarOrdering),
        getVarOrderingHeuristic(addVarOrderingHeuristic, inverseAddVarOrdering)
    )

    return heuristicTuple

def getNamedHeuristicTuple(heuristicTuple):
    assert len(heuristicTuple) == 3

    def getNamedClusteringHeuristic(clusteringHeuristic):
        return ClusteringHeuristic(clusteringHeuristic).name

    def getNamedVarOrderingHeuristic(varOrderingHeuristic):
        return VarOrderingHeuristic(varOrderingHeuristic.replace('-', '')).name

    def getInverseVarOrderingStr(varOrderingHeuristic):
        return str(int(varOrderingHeuristic.startswith('-')))

    (clusteringHeuristic, formulaVarOrderingHeuristic, addVarOrderingHeuristic) = heuristicTuple
    return (
        getNamedClusteringHeuristic(clusteringHeuristic),
        getNamedVarOrderingHeuristic(formulaVarOrderingHeuristic),
        getInverseVarOrderingStr(formulaVarOrderingHeuristic),
        getNamedVarOrderingHeuristic(addVarOrderingHeuristic),
        getInverseVarOrderingStr(addVarOrderingHeuristic)
    )

def getShortHeuristicName(heuristicStr):
    assert type(heuristicStr) == str

    try:
        clusteringHeuristic = ClusteringHeuristic[heuristicStr]
        return CLUSTERING_HEURISTIC_SHORT_NAMES[clusteringHeuristic]
    except Exception:
        return heuristicStr

def getSublists(vector, sublistLen): # [a, b, c, d,...] |-> [[a, b], [c, d],...]
    sublists = [vector[i : i + sublistLen] for i in range(0, len(vector), sublistLen)]
    return sublists

def mergeSublists(sublists): # [[a, b], [c, d],...] |-> [a, b, c, d,...]
    return [item for sublist in sublists for item in sublist]

################################################################################
'''Handles errors.'''

class ExperimentException(Exception):
    pass

def printSeparator(char='%'):
    print(char * 70)

def emphasize(message):
    return '{}\n'.format(message)

def raiseException(message, suppressed=False):
    message = emphasize(message)
    if suppressed:
        if SHOWING_WARNINGS:
            print('SUPPRESSED EXCEPTION: {}'.format(message))
    else:
        raise ExperimentException(message)

def raiseWrongExperimentException(experiment, suppressed=False):
    raiseException('wrong Experiment: {}'.format(experiment), suppressed)

def raiseWrongCounterException(counter, suppressed=False):
    raiseException('wrong Counter: {}'.format(counter), suppressed)

def raiseWrongWeightFormatException(weightFormat, suppressed=False):
    raiseException('wrong WeightFormat: {}'.format(weightFormat), suppressed)

def raiseWrongBenchmarkFamException(benchmarkFam, suppressed=False):
    raiseException('wrong benchmark family:\n\t{}'.format(benchmarkFam), suppressed)

def raiseMissingCnfFileNameException(outFilePath, suppressed=False):
    raiseException('missing CNF file name for OUT file:\n\t{}'.format(outFilePath), suppressed)

def raiseMissingOutFileStatusException(outFilePath, suppressed=False):
    raiseException('missing status for OUT file:\n\t{}'.format(outFilePath), suppressed)

def raiseMissingModelCountException(outFilePath, suppressed=False):
    raiseException('missing model count for OUT file:\n\t{}'.format(outFilePath), suppressed)

def showWarning(message):
    if SHOWING_WARNINGS:
        print(emphasize(message))

def showUnusualOutFileStatusWarning(outFileStatus, outFilePath):
    showWarning(emphasize('WARNING: unusual status ({}) for OUT file:\n\t{}'.format(outFileStatus.name, outFilePath)))

def showSpaceoutOutFileStatusWarning(outFilePath):
    showUnusualOutFileStatusWarning(OutFileStatus.SPACEOUT, outFilePath)

def showKillOutFileStatusWarning(outFilePath):
    showUnusualOutFileStatusWarning(OutFileStatus.KILL, outFilePath)

def showErrorOutFileStatusWarning(outFilePath):
    showUnusualOutFileStatusWarning(OutFileStatus.ERROR, outFilePath)

debug = print

################################################################################
'''Handles configs.'''

NO_LIN_CLUSTERING_CONFIGS = [
    ClusteringHeuristic.MONOLITHIC,
    # ClusteringHeuristic.LINEAR, # ignored in paper
    ClusteringHeuristic.BUCKET_LIST,
    ClusteringHeuristic.BUCKET_TREE,
    ClusteringHeuristic.BOUQUET_LIST,
    ClusteringHeuristic.BOUQUET_TREE
]

## altogether/exp1:
BEST_HEURISTIC_TUPLE = getHeuristicTuple('BOUQUET_TREE LEXP 0 MCS 0')
BEST_HEURISTIC_TUPLE_2 = getHeuristicTuple('BUCKET_TREE LEXP 1 MCS 0')
BEST_HEURISTIC_TUPLE_3 = getHeuristicTuple('BOUQUET_LIST LEXP 0 MCS 0')
BEST_HEURISTIC_TUPLE_4 = getHeuristicTuple('BOUQUET_TREE LEXP 0 MCS 1')
BEST_HEURISTIC_TUPLE_5 = getHeuristicTuple('BOUQUET_LIST LEXP 0 MCS 1')
BEST_HEURISTIC_TUPLES = [BEST_HEURISTIC_TUPLE, BEST_HEURISTIC_TUPLE_2, BEST_HEURISTIC_TUPLE_3, BEST_HEURISTIC_TUPLE_4, BEST_HEURISTIC_TUPLE_5]

NO_LIN_MEDIAN_HEURISTIC_TUPLE = getHeuristicTuple('BUCKET_LIST LEXP 0 LEXP 0')
BEST_HEURISTIC_TUPLE_MONO = getHeuristicTuple('MONOLITHIC MCS 1 LEXP 0')
WORST_HEURISTIC_TUPLE = getHeuristicTuple('BUCKET_LIST RANDOM 0 RANDOM 0')

def getHeuristicTuples(experiment):
    if experiment == Experiment.ALL_CONFIGS:
        heuristicTuples = itertools.product(getClusteringHeuristics(), getVarOrderingHeuristics(), getVarOrderingHeuristics())
        heuristicTuples = sorted(heuristicTuples)
        return heuristicTuples
    elif experiment == Experiment.BEST_CONFIGS:
        return BEST_HEURISTIC_TUPLES
    elif experiment == Experiment.COUNTERS:
        return [BEST_HEURISTIC_TUPLE]
    else:
        raiseWrongExperimentException(experiment)

################################################################################
'''Generates commands.'''

def getWrappedCommands(commands):
    command = ' && '.join(commands)
    return 'wrapper.py "{}"'.format(command)

def getAddmcCommand(cnfFilePath, weightFormat, heuristicTuple):
    (clusteringHeuristic, formulaVarOrderingHeuristic, addVarOrderingHeuristic) = heuristicTuple
    return ' '.join([
        'addmc',
        '--cf', cnfFilePath,
        '--wf', weightFormat.value,
        '--ch', clusteringHeuristic,
        '--cv', formulaVarOrderingHeuristic,
        '--dv', addVarOrderingHeuristic,
    ])

def getCommands(experiment, counter, weightFormat, benchmarkFam, heuristicTuples=[]):
    commands = []
    for cnfFilePath in getCnfFilePaths(benchmarkFam, weightFormat, counter):
        if counter == Counter.ADDMC:
            if not heuristicTuples:
                heuristicTuples = getHeuristicTuples(experiment)
            for heuristicTuple in heuristicTuples:
                command = getAddmcCommand(cnfFilePath, weightFormat, heuristicTuple)
                command = getWrappedCommands([command])
                commands.append(command)
        elif counter == Counter.CACHET:
            command = 'cachet {}'.format(cnfFilePath)
            command = getWrappedCommands([command])
            commands.append(command)
        elif counter == Counter.MINIC2D:
            command = '~/bin/miniC2D -W' # unweighted: -C -i
            command = '{} -c {}'.format(command, cnfFilePath)
            command = getContainedCommand(command)
            command = getWrappedCommands([command])
            commands.append(command)
        elif counter == Counter.C2D:
            nnfFilePath = getNnfFilePath(cnfFilePath) # nnfFile will be written when C2D is called
            queryFilePath = getQueryFilePath(cnfFilePath) # queryFile is written by function getQueryFilePath

            c2dCommand = '~/bin/c2d -reduce'.format(cnfFilePath) # unweighted: -count -in_memory -reduce
            c2dCommand = '{} -in {}'.format(c2dCommand, cnfFilePath)
            c2dCommand = getContainedCommand(c2dCommand)

            queryCommand = 'query-dnnf -cmd {}'.format(queryFilePath)
            command = getWrappedCommands([c2dCommand, queryCommand])

            #NOTE should find a way to remove NNF file even when query-dnnf times out:
            # rmCommand = 'rm -f {} {}'.format(nnfFilePath, queryFilePath)
            # command = '{} && {}'.format(command, rmCommand)

            commands.append(command)
        elif counter == Counter.D4:
            command = 'd4 {}'.format(cnfFilePath)
            if weightFormat == WeightFormat.DDNNF:
                command = command + ' -wFile={}'.format(getWeightFilePath(cnfFilePath))
            command = getWrappedCommands([command])
            commands.append(command)
        elif counter == Counter.SHARPSAT:
            command = 'sharpSAT {}'.format(cnfFilePath)
            command = getWrappedCommands([command])
            commands.append(command)
        else:
            raiseWrongCounterException(counter)

    return commands

################################################################################
'''Processes SLURM array commands.'''

## computing cluster's constants:
SLURM_SUBARRAY_LEN_CAP = 1000 # SLURM_ARRAY_TASK_MAX
SLURM_ARRAY_LEN_CAP = 10000 # MaxArraySize

class CommandSpec:
    def __init__(self, experiment, counter, weightFormat, benchmarkFam, heuristicTuples):
        self.experiment = experiment
        self.counter = counter
        self.weightFormat = weightFormat
        self.benchmarkFam = benchmarkFam
        self.heuristicTuples = heuristicTuples

    def getCommands(self):
        return getCommands(self.experiment, self.counter, self.weightFormat, self.benchmarkFam, self.heuristicTuples)

class SlurmArrayRun:
    def __init__(self, commandSpecs, commandTimeout, commandsPerJob):
        randNum = random.randrange(1e3)
        print('\nRandom SLURM array number: {}'.format(randNum))
        self.arrayId = randNum
        self.commandSpecs = commandSpecs
        self.commandTimeout = commandTimeout
        self.commandsPerJob = commandsPerJob

    def getScriptsPath(self):
        return os.path.join(EVAL_PATH, 'AutoScripts-{}'.format(self.arrayId))

    def getCountingScriptsPath(self):
        return os.path.join(self.getScriptsPath(), 'counting')

    def getSubmittingScriptsPath(self):
        return os.path.join(self.getScriptsPath(), 'submitting')

    def getOutputsPath(self):
        return os.path.join(EVAL_PATH, 'AutoOutputs-{}'.format(self.arrayId))

    def getRedirectionOutputsPath(self):
        return os.path.join(self.getOutputsPath(), 'redirection')

    def getSlurmOutputsPath(self):
        return os.path.join(self.getOutputsPath(), 'slurm')

    def getSubmissionLogPath(self):
        return os.path.join(EVAL_PATH, 'AutoSubmissionLog-{}.txt'.format(self.arrayId))

    def writeCountingScriptFiles(self): # returns SLURM array length
        commands = []
        for commandSpec in self.commandSpecs:
            commands.extend(commandSpec.getCommands())

        jobs = getSublists(commands, self.commandsPerJob)
        jobCount = len(jobs)
        if jobCount > SLURM_ARRAY_LEN_CAP:
            raiseException('jobCount = {} > MaxArraySize = {}'.format(jobCount, SLURM_ARRAY_LEN_CAP))
        elif jobCount > 0:
            os.makedirs(self.getCountingScriptsPath())
            os.makedirs(self.getSubmittingScriptsPath())
            os.makedirs(self.getRedirectionOutputsPath())
            for slurmArrayTaskId in range(jobCount):
                commands = jobs[slurmArrayTaskId]
                countingScriptPath = os.path.join(self.getCountingScriptsPath(), 'c{}.sh'.format(slurmArrayTaskId))
                with open(countingScriptPath, 'w') as countingScript:
                    for i in range(len(commands)):
                        command = commands[i]
                        redirectionOutputPath = os.path.join(self.getRedirectionOutputsPath(), 'o{}_{}.out'.format(slurmArrayTaskId, i))
                        command = '{} &> {}'.format(command, redirectionOutputPath)
                        command = 'timeout {}s {}'.format(self.commandTimeout, command)
                        countingScript.write('\n{}\n'.format(command))
                os.chmod(countingScriptPath, stat.S_IRUSR ^ stat.S_IWUSR ^ stat.S_IXUSR) # owner reads/writes/executes
        return jobCount

    def writeSubmittingScriptFiles(self):
        slurmArrayLen = self.writeCountingScriptFiles()
        for i in range(0, slurmArrayLen, SLURM_SUBARRAY_LEN_CAP):
            submittingScriptIndex = i // SLURM_SUBARRAY_LEN_CAP
            slurmOutputsPath = os.path.join(self.getSlurmOutputsPath(), str(submittingScriptIndex))
            slurmArrayCommand = 'mkdir -p {} && '.format(slurmOutputsPath) + ' '.join([
                'sbatch',
                '--output={}/%A_%a.out'.format(slurmOutputsPath),
                '--array={}-{}'.format(i, min(i + SLURM_SUBARRAY_LEN_CAP, slurmArrayLen) - 1),
                '--export=countingScriptsPath={},ALL'.format(self.getCountingScriptsPath()),
                SLURM_ARRAY_SBATCH_PATH,
            ])

            submittingScriptPath = os.path.join(self.getSubmittingScriptsPath(), 's{}.sh'.format(submittingScriptIndex))
            with open(submittingScriptPath, 'w') as submittingScript:
                submittingScript.write('\n{}\n\n'.format(slurmArrayCommand))
                os.chmod(submittingScriptPath, stat.S_IRUSR ^ stat.S_IWUSR ^ stat.S_IXUSR) # owner reads/writes/executes

    def getSubmittingScriptPaths(self):
        return getFilePaths(self.getSubmittingScriptsPath(), '.sh')

    def logSubmissionEvent(self, message):
        print(message)

        with open(self.getSubmissionLogPath(), 'a') as submissionLog:
            submissionLog.write(message + '\n')

    def attemptSubmission(self):
        submittingScriptPaths = self.getSubmittingScriptPaths()
        submittingScriptPaths = sorted(submittingScriptPaths, key=(lambda filePath: int(getBaseFileName(filePath)[1:]))) # '{dir}/s0.sh' |-> 0
        self.logSubmissionEvent('\nSubmission scripts remaining: {}'.format(len(submittingScriptPaths)))
        if submittingScriptPaths:
            submittingScriptPath = submittingScriptPaths[0]
            self.logSubmissionEvent('Submission script: {}\nSubmission attempt: {}'.format(submittingScriptPath, getTime()))

            command = 'bash {}'.format(submittingScriptPath)
            with open(self.getSubmissionLogPath(), 'a') as submissionLog:
                returnCode = subprocess.call(
                    command.split(),
                    stdout=submissionLog,
                    stderr=submissionLog
                )
                if int(returnCode) == 0:
                    self.logSubmissionEvent('Submission succeeded')
                    os.rename(submittingScriptPath, submittingScriptPath.replace('.sh', '.txt'))
                else:
                    self.logSubmissionEvent('Submission failed')

    def runSubmissionLoop(self):
        loopDurationCap = INF # seconds
        submissionAttemptInterval = 60 # seconds

        loopStartTime = getTime()
        while getDuration(loopStartTime) <= loopDurationCap and self.getSubmittingScriptPaths():
            self.attemptSubmission()
            time.sleep(submissionAttemptInterval)

        self.logSubmissionEvent('\nAll submissions completed: no submission script remains\n')

    def runArray(self):
        self.writeSubmittingScriptFiles()
        self.runSubmissionLoop()

def mainSlurmArray():
    commandSpecs = [
        # CommandSpec(Experiment.ALL_CONFIGS, Counter.ADDMC, WeightFormat.MINIC2D, BenchmarkFam.ALTOGETHER, []),
        # CommandSpec(Experiment.COUNTERS, Counter.ADDMC, WeightFormat.MINIC2D, BenchmarkFam.ALTOGETHER, []),
        # CommandSpec(Experiment.COUNTERS, Counter.CACHET, WeightFormat.CACHET, BenchmarkFam.ALTOGETHER, []),
        # CommandSpec(Experiment.COUNTERS, Counter.D4, WeightFormat.DDNNF, BenchmarkFam.ALTOGETHER, []),
        # CommandSpec(Experiment.COUNTERS, Counter.MINIC2D, WeightFormat.MINIC2D, BenchmarkFam.ALTOGETHER, []),
        # CommandSpec(Experiment.COUNTERS, Counter.C2D, WeightFormat.DDNNF, BenchmarkFam.ALTOGETHER, []), #NOTE rm *nnf *query
    ]

    #NOTE must edit SlurmArray.batch first:
    commandTimeout = 1000
    commandsPerJob = 10

    slurmArrayRun = SlurmArrayRun(commandSpecs, commandTimeout, commandsPerJob)
    slurmArrayRun.runArray()

################################################################################
'''Creates OUT file reports.'''

def getContainedCommand(command):
    return 'singularity exec $HOME/bin/centos.simg {}'.format(command)

def getUncontainedCommand(command):
    if command.startswith('singularity exec'):
        return ' '.join(command.split()[3:])
    else:
        return command

def isWrapperLine(line):
    return line.strip().startswith('[WRAPPER]')

def getWrapperLineValue(line):
    assert isWrapperLine(line)
    return ' '.join(line.split()[2:])

def isWrapperCommandLine(line):
    return isWrapperLine(line) and line.split()[1] == 'command'

def isWrapperDurationLine(line):
    return isWrapperLine(line) and line.split()[1] == 'duration_seconds'

def getWrapperDuration(line):
    assert isWrapperDurationLine(line)
    return float(getWrapperLineValue(line))

def isEchoCnfFilePathLine(line):
    words = line.split()
    return len(words) == 2 and words[0] == 'cnfFilePath'

def getCnfFileName(line):
    assert isEchoCnfFilePathLine(line)
    return getFileName(line.split()[1])

def isTimeoutLine(line):
    return line.strip().endswith('DUE TO TIME LIMIT ***')

def isSpaceoutLine(line):
    return line.strip() == 'slurmstepd: Exceeded step memory limit at some point.'

def isKillLine(line):
    return 'Killed' in line

def getAddmcOutFileReport(outFilePath, experiment):
    def isWrapperHeuristicTupleLine(line):
        return isWrapperCommandLine(line) and getWrapperLineValue(line).split()[0] == 'addmc'

    def isErrorLine(line):
        return line.startswith('ERROR:')

    outFileStatus = None
    modelCount = None
    overflow = False
    time = INF

    heuristicTuple = None # with wrapper.py

    ## without wrapper.py:
    namedClusteringHeuristic = None
    namedFormulaVarOrderingHeuristic = None
    inverseFormulaVarOrdering = None
    namedAddVarOrderingHeuristic = None
    inverseAddVarOrdering = None

    mavc = None

    with open(outFilePath) as outFile:
        for line in outFile:
            line = line.strip()
            if isWrapperHeuristicTupleLine(line):
                words = tuple(getWrapperLineValue(line).split())
                if '--cf' in words:
                    heuristicTuple = words[-5::2]
                else:
                    heuristicTuple = words[-3:]
            elif isTimeoutLine(line):
                outFileStatus = OutFileStatus.TIMEOUT
                break
            elif isSpaceoutLine(line):
                showSpaceoutOutFileStatusWarning(outFilePath)
                outFileStatus = OutFileStatus.SPACEOUT
                break
            elif isKillLine(line):
                showKillOutFileStatusWarning(outFilePath)
                outFileStatus = OutFileStatus.KILL
                break
            elif isErrorLine(line):
                showErrorOutFileStatusWarning(outFilePath)
                outFileStatus = OutFileStatus.ERROR
                break
            elif line:
                words = line.split()
                if words[0] == '*':
                    (key, value) = words[1:]
                    if key == 'cnfFilePath':
                        cnfFileName = getFileName(value)
                    elif key == 'weightFormat':
                        pass
                    elif key == 'clusterPartition':
                        namedClusteringHeuristic = value
                    elif key == 'clusterVarOrder':
                        namedFormulaVarOrderingHeuristic = value
                    elif key == 'inverseClusterVarOrder':
                        inverseFormulaVarOrdering = value
                    elif key == 'diagramVarOrder':
                        namedAddVarOrderingHeuristic = value
                    elif key == 'inverseDiagramVarOrder':
                        inverseAddVarOrdering = value
                    elif key == 'maxAddVarCount':
                        mavc = int(value)
                    elif key == 'modelCount':
                        modelCount = float(value)
                        overflow = math.isinf(modelCount)
                    elif key == 'seconds':
                        time = float(value)
                        outFileStatus = OutFileStatus.COMPLETION if time <= getTimeLimit(experiment) else OutFileStatus.TIMEOUT
                    else:
                        pass
                        # raise ExperimentException('wrong ADDMC "*"-line in file {}'.format(outFilePath))
    if outFileStatus == None:
        # raiseMissingOutFileStatusException(outFilePath)
        outFileStatus = OutFileStatus.TIMEOUT

    if heuristicTuple == None:
        namedHeuristicTuple = (namedClusteringHeuristic, namedFormulaVarOrderingHeuristic, inverseFormulaVarOrdering, namedAddVarOrderingHeuristic, inverseAddVarOrdering)
        if None in namedHeuristicTuple:
            showError('missing heuristic for OUT file:\n\t{}'.format(outFilePath))
        heuristicTuple = getHeuristicTuple(namedHeuristicTuple)

    return AddmcOutFileReport(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, heuristicTuple, mavc, experiment)

def getCachetOutFileReport(outFilePath, needsScaling):
    outFileStatus = None
    modelCount = None
    overflow = False
    time = INF
    for line in open(outFilePath):
        line = line.strip()
        if isTimeoutLine(line):
            outFileStatus = OutFileStatus.TIMEOUT
            break
        elif line:
            words = line.split()
            if words[0] == 'Solving':
                cnfFileName = getFileName(words[1])
            elif words[0] == 'Total':
                time = float(words[-1])
                outFileStatus = OutFileStatus.COMPLETION if time <= getTimeLimit(Experiment.COUNTERS) else OutFileStatus.TIMEOUT
            elif words[0] == 'Satisfying':
                modelCount = float(words[-1])
                assert not math.isnan(modelCount)
                if math.isinf(modelCount):
                    showWarning('Cachet\'s 80-bit float may have overflowed when being converted to 64-bit float in OUT file {}'.format(outFilePath))
                    modelCount = DUMMY_MODEL_COUNT
                    global globalCachetDummyModelCounts1
                    globalCachetDummyModelCounts1.append(outFilePath)
                elif needsScaling:
                    base = 2. # CACHET's weights: (.25, .75), other counters' weights: (.5, 1.5)
                    exponent = PSEUDOWEIGHTED_VAR_COUNTS[cnfFileName]
                    try:
                        scale = base ** exponent # may raise OverflowError
                        modelCount *= scale
                    except OverflowError:
                        modelCount = DUMMY_MODEL_COUNT
                        global globalCachetDummyModelCounts2
                        globalCachetDummyModelCounts2.append(outFilePath)
                    assert math.isfinite(modelCount)
    if outFileStatus == None:
        # raiseMissingOutFileStatusException(outFilePath)
        outFileStatus = OutFileStatus.TIMEOUT
    return CachetOutFileReport(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time)

def getMinic2dOutFileReport(outFilePath):
    def isWrapperCnfFilePathLine(line):
        return isWrapperCommandLine(line) and getUncontainedCommand(getWrapperLineValue(line)).startswith('~/bin/miniC2D')

    def isErrorLine(line):
        line = line.strip()
        return line == 'Fewer clauses in cnf file than declared!' or line.endswith('exceeds max length of 100000!')

    def isWeightedCountLine(line):
        words = line.split()
        return len(words) > 1 and words[0] == 'Count' and 'stats:' != words[1] != 'Time'

    def isUnweightedCountLine(line):
        words = line.split()
        return len(words) > 1 and words[0] == 'Counting...' and words[1].isdigit()

    cnfFileName = None
    outFileStatus = None
    modelCount = None
    overflow = False
    time = INF
    for line in open(outFilePath):
        line = line.strip()
        if isEchoCnfFilePathLine(line):
            cnfFileName = getCnfFileName(line)
        elif isWrapperCnfFilePathLine(line):
            c2dCommand = getUncontainedCommand(getWrapperLineValue(line))
            cnfFilePath = c2dCommand.split()[-1] # '~/bin/miniC2D (-W | (-C -i)) -c {cnfFilePath}' |-> '{cnfFilePath}'
            cnfFileName = getFileName(cnfFilePath)
        elif isWrapperDurationLine(line):
            if modelCount == None:
                showErrorOutFileStatusWarning(outFilePath)
                outFileStatus = OutFileStatus.ERROR
                break
            else:
                time = getWrapperDuration(line)
                outFileStatus = OutFileStatus.COMPLETION if time <= getTimeLimit(Experiment.COUNTERS) else OutFileStatus.TIMEOUT
        elif isWeightedCountLine(line):
            modelCount = float(line.split()[1])
        elif isUnweightedCountLine(line):
            modelCount = float(line.split()[1])
        elif isTimeoutLine(line):
            outFileStatus = OutFileStatus.TIMEOUT
            break
        elif isErrorLine(line):
            showErrorOutFileStatusWarning(outFilePath)
            outFileStatus = OutFileStatus.ERROR
            break
        elif line:
            words = line.split()
            if words[0] == 'cnfFilePath':
                cnfFileName = getFileName(words[1])
            elif words[0] == 'Total':
                time = float(words[-1].replace('s', ''))
                outFileStatus = OutFileStatus.COMPLETION if time <= getTimeLimit(Experiment.COUNTERS) else OutFileStatus.TIMEOUT
    if outFileStatus == None:
        # raiseMissingOutFileStatusException(outFilePath)
        outFileStatus = OutFileStatus.TIMEOUT
    return Minic2dOutFileReport(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time)

def getC2dOutFileReport(outFilePath):
    def isWrapperCnfFilePathLine(line):
        return isWrapperCommandLine(line) and getUncontainedCommand(getWrapperLineValue(line)).startswith('~/bin/c2d')

    def isDdnnfCountLine(line):
        return line.startswith('> >') and len(line) > 4 and not isTimeoutLine(line)

    def isC2dCountLine(line):
        return line.startswith('Counting...')

    def isErrorLine(line):
        return line.endswith('exceeds max length of 100000!')

    cnfFileName = None
    outFileStatus = None
    modelCount = None
    overflow = False
    time = INF
    compilerTime = INF
    for line in open(outFilePath):
        line = line.strip()
        if isWrapperCnfFilePathLine(line):
            c2dCommand = getUncontainedCommand(getWrapperLineValue(line))
            cnfFilePath = c2dCommand.split()[-1] # '~/bin/c2d [-count -in_memory] -reduce -in {cnfFilePath}' |-> '{cnfFilePath}'
            cnfFileName = getFileName(cnfFilePath)
        elif isWrapperDurationLine(line): # weighted model counting with d-dnnf-reasoner
            if modelCount == None:
                showErrorOutFileStatusWarning(outFilePath)
                outFileStatus = OutFileStatus.ERROR
                break
            else: # may overwrite `time` and `outFileStatus`
                time = getWrapperDuration(line)
                outFileStatus = OutFileStatus.COMPLETION if time <= getTimeLimit(Experiment.COUNTERS) else OutFileStatus.TIMEOUT
        elif line.startswith('Total Time:'): # `time` and `outFileStatus` may be overwritten later
            time = compilerTime = float(line.split()[-1][:-1]) # line == 'Total Time: 32.480s'
            outFileStatus = OutFileStatus.COMPLETION if compilerTime <= getTimeLimit(Experiment.COUNTERS) else OutFileStatus.TIMEOUT
        elif isDdnnfCountLine(line):
            modelCount = float(line.split()[2])
        elif isC2dCountLine(line):
            modelCount = float(line.split()[0].replace('Counting...', ''))
        elif isTimeoutLine(line):
            outFileStatus = OutFileStatus.TIMEOUT
            break
        elif isSpaceoutLine(line):
            showSpaceoutOutFileStatusWarning(outFilePath)
            outFileStatus = OutFileStatus.SPACEOUT
            break
        elif isErrorLine(line):
            showErrorOutFileStatusWarning(outFilePath)
            outFileStatus = OutFileStatus.ERROR
            break
    if cnfFileName == None:
        raiseMissingCnfFileNameException(outFilePath)
    if outFileStatus == None:
        # raiseMissingOutFileStatusException(outFilePath)
        outFileStatus = OutFileStatus.TIMEOUT
    if modelCount == None:
        # raiseMissingModelCountException(outFilePath)
        outFileStatus = OutFileStatus.TIMEOUT
    return C2dOutFileReport(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time, compilerTime)

def getD4OutFileReport(outFilePath):
    def isWrapperCnfFilePathLine(line):
        return isWrapperCommandLine(line) and getWrapperLineValue(line).split()[0] == 'd4'

    def isErrorLine(line):
        return line.strip() == 'PARSE ERROR! Unexpected char: .'

    cnfFileName = None
    outFileStatus = None
    modelCount = None
    overflow = False
    time = INF
    for line in open(outFilePath):
        line = line.strip()
        if isEchoCnfFilePathLine(line):
            cnfFileName = getCnfFileName(line)
        elif isWrapperCnfFilePathLine(line):
            cnfFileName = getFileName(getWrapperLineValue(line).split()[1])
        elif isWrapperDurationLine(line):
            time = getWrapperDuration(line)
            outFileStatus = OutFileStatus.COMPLETION if time <= getTimeLimit(Experiment.COUNTERS) else OutFileStatus.TIMEOUT
        elif isTimeoutLine(line):
            outFileStatus = OutFileStatus.TIMEOUT
            break
        elif isErrorLine(line):
            showErrorOutFileStatusWarning(outFilePath)
            outFileStatus = OutFileStatus.ERROR
            break
        elif line:
            words = line.split()
            if line.startswith('c Final time:'):
                time = float(words[-1])
                outFileStatus = OutFileStatus.COMPLETION if time <= getTimeLimit(Experiment.COUNTERS) else OutFileStatus.TIMEOUT
            elif words[0] == 's':
                modelCount = float(words[1])
    if outFileStatus == None:
        # raiseMissingOutFileStatusException(outFilePath)
        outFileStatus = OutFileStatus.TIMEOUT
    return D4OutFileReport(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time)

def getSharpsatOutFileReport(outFilePath):
    def isErrorLine(line):
        return line.endswith('EMPTY CLAUSE FOUND') or 'Segmentation fault' in line

    outFileStatus = None
    modelCount = None
    overflow = False
    time = INF
    for line in open(outFilePath):
        line = line.strip()
        if isWrapperDurationLine(line):
            time = getWrapperDuration(line)
            outFileStatus = OutFileStatus.COMPLETION if time <= getTimeLimit(Experiment.COUNTERS) else OutFileStatus.TIMEOUT
        elif isTimeoutLine(line):
            outFileStatus = OutFileStatus.TIMEOUT
            break
        elif isSpaceoutLine(line):
            showSpaceoutOutFileStatusWarning(outFilePath)
            outFileStatus = OutFileStatus.SPACEOUT
            break
        elif isErrorLine(line):
            showErrorOutFileStatusWarning(outFilePath)
            outFileStatus = OutFileStatus.ERROR
            break
        elif line:
            words = line.split()
            if words[0] == 'Solving':
                cnfFileName = getFileName(words[1])
            elif words[0] == 'time:':
                time = float(words[-1].replace('s', ''))
                outFileStatus = OutFileStatus.COMPLETION if time <= getTimeLimit(Experiment.COUNTERS) else OutFileStatus.TIMEOUT
            elif len(words) == 1:
                word = words[0]
                if word.isdigit():
                    modelCount = float(word)
    if outFileStatus == None:
        raiseMissingOutFileStatusException(outFilePath)
    return SharpsatOutFileReport(outFilePath, outFileStatus, cnfFileName, modelCount, overflow, time)

def getOutFileReport(counter, outFilePath, experiment, needsScaling):
    if counter == Counter.ADDMC:
        return getAddmcOutFileReport(outFilePath, experiment)
    elif counter == Counter.CACHET:
        return getCachetOutFileReport(outFilePath, needsScaling)
    elif counter == Counter.MINIC2D:
        return getMinic2dOutFileReport(outFilePath)
    elif counter == Counter.C2D:
        return getC2dOutFileReport(outFilePath)
    elif counter == Counter.D4:
        return getD4OutFileReport(outFilePath)
    elif counter == Counter.SHARPSAT:
        return getSharpsatOutFileReport(outFilePath)
    else:
        raiseWrongCounterException(counter)

def getOutFileReports(experiment, counter, benchmarkFam, weightFormat, heuristicTuple=None, clusteringHeuristics=[]):
    assert experiment in Experiment
    assert counter in Counter
    assert benchmarkFam in BenchmarkFam
    assert weightFormat in WeightFormat
    assert heuristicTuple == None or len(heuristicTuple) == 3
    assert all([clusteringHeuristic in ClusteringHeuristic for clusteringHeuristic in clusteringHeuristics])

    if weightFormat == WeightFormat.WEIGHTED:
        weightFormat = WeightFormat.getSpecificWeightedFormat(counter)

    if experiment == Experiment.COUNTERS and counter == Counter.ADDMC and heuristicTuple == None:
        heuristicTuple = BEST_HEURISTIC_TUPLE

    def computeOutFileReports():
        needsScaling = counter == Counter.CACHET and benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED
        outFilePaths = getOutFilePaths(experiment, counter, weightFormat, benchmarkFam=benchmarkFam)
        outFileReports = [
            getOutFileReport(counter, outFilePath, experiment, needsScaling)
            for outFilePath in outFilePaths
        ]
        return outFileReports

    def filterOutFileReports(outFileReports):
        localCopy = outFileReports.copy()

        if heuristicTuple:
            localCopy = [
                outFileReport for outFileReport in localCopy
                if outFileReport.heuristicTuple == heuristicTuple
            ]

        if clusteringHeuristics:
            localCopy = [
                outFileReport for outFileReport in localCopy
                if ClusteringHeuristic(outFileReport.heuristicTuple[0]) in clusteringHeuristics
            ]

        return localCopy

    if benchmarkFam == BenchmarkFam.ALTOGETHER:
        assert weightFormat.isWeighted()
        bayesOutFileReports = getOutFileReports(experiment, counter, BenchmarkFam.BAYES, WeightFormat.WEIGHTED, heuristicTuple, clusteringHeuristics)
        pseudoweightedOutFileReports = getOutFileReports(experiment, counter, BenchmarkFam.PSEUDOWEIGHTED, WeightFormat.WEIGHTED, heuristicTuple, clusteringHeuristics)
        outFileReports = bayesOutFileReports + pseudoweightedOutFileReports
        return outFileReports

    if counter == Counter.ADDMC:
        if experiment == Experiment.ALL_CONFIGS:
            if benchmarkFam == BenchmarkFam.BAYES:
                global globalReportsExp1AddmcBayesMinic2d
                if WRITING_DATA_TO_PY_FILE:
                    globalReportsExp1AddmcBayesMinic2d = computeOutFileReports()
                elif not globalReportsExp1AddmcBayesMinic2d:
                    from data import reportsExp1AddmcBayesMinic2d
                    globalReportsExp1AddmcBayesMinic2d = reportsExp1AddmcBayesMinic2d
                return filterOutFileReports(globalReportsExp1AddmcBayesMinic2d)
            elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
                global globalReportsExp1AddmcPseudoweightedMinic2d
                if WRITING_DATA_TO_PY_FILE:
                    globalReportsExp1AddmcPseudoweightedMinic2d = computeOutFileReports()
                elif not globalReportsExp1AddmcPseudoweightedMinic2d:
                    from data import reportsExp1AddmcPseudoweightedMinic2d
                    globalReportsExp1AddmcPseudoweightedMinic2d = reportsExp1AddmcPseudoweightedMinic2d
                return filterOutFileReports(globalReportsExp1AddmcPseudoweightedMinic2d)
            else:
                raiseWrongBenchmarkFamException(benchmarkFam)
        elif experiment == Experiment.BEST_CONFIGS:
            if benchmarkFam == BenchmarkFam.BAYES:
                global globalReportsExp1BAddmcBayesMinic2d
                if not globalReportsExp1BAddmcBayesMinic2d:
                    globalReportsExp1BAddmcBayesMinic2d = computeOutFileReports()
                return filterOutFileReports(globalReportsExp1BAddmcBayesMinic2d)
            elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
                global globalReportsExp1BAddmcPseudoweightedMinic2d
                if not globalReportsExp1BAddmcPseudoweightedMinic2d:
                    globalReportsExp1BAddmcPseudoweightedMinic2d = computeOutFileReports()
                return filterOutFileReports(globalReportsExp1BAddmcPseudoweightedMinic2d)
            else:
                raiseWrongBenchmarkFamException(benchmarkFam)
        elif experiment == Experiment.COUNTERS:
            if benchmarkFam == BenchmarkFam.BAYES:
                global globalReportsExp2AddmcBayesMinic2d
                if not globalReportsExp2AddmcBayesMinic2d:
                    globalReportsExp2AddmcBayesMinic2d = computeOutFileReports()
                return filterOutFileReports(globalReportsExp2AddmcBayesMinic2d)
            elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
                global globalReportsExp2AddmcPseudoweightedMinic2d
                if not globalReportsExp2AddmcPseudoweightedMinic2d:
                    globalReportsExp2AddmcPseudoweightedMinic2d = computeOutFileReports()
                return filterOutFileReports(globalReportsExp2AddmcPseudoweightedMinic2d)
            else:
                raiseWrongBenchmarkFamException(benchmarkFam)
        else:
            raiseWrongExperimentException(experiment)
    elif counter == Counter.CACHET:
        if benchmarkFam == BenchmarkFam.BAYES:
            if weightFormat == WeightFormat.CACHET:
                global globalReportsExp2CachetBayesCachet
                if not globalReportsExp2CachetBayesCachet:
                    globalReportsExp2CachetBayesCachet = computeOutFileReports()
                return globalReportsExp2CachetBayesCachet
            else:
                raise raiseWrongWeightFormatException(weightFormat)
        elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
            if weightFormat == WeightFormat.CACHET:
                global globalReportsExp2CachetPseudoweightedCachet
                if not globalReportsExp2CachetPseudoweightedCachet:
                    globalReportsExp2CachetPseudoweightedCachet = computeOutFileReports()
                return globalReportsExp2CachetPseudoweightedCachet
            else:
                raise raiseWrongWeightFormatException(weightFormat)
        else:
            raiseWrongBenchmarkFamException(benchmarkFam)
    elif counter == Counter.MINIC2D:
        if benchmarkFam == BenchmarkFam.BAYES:
            if weightFormat == WeightFormat.MINIC2D:
                global globalReportsExp2Minic2dBayesMinic2d
                if not globalReportsExp2Minic2dBayesMinic2d:
                    globalReportsExp2Minic2dBayesMinic2d = computeOutFileReports()
                return globalReportsExp2Minic2dBayesMinic2d
            else:
                raise raiseWrongWeightFormatException(weightFormat)
        elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
            if weightFormat == WeightFormat.MINIC2D:
                global globalReportsExp2Minic2dPseudoweightedMinic2d
                if not globalReportsExp2Minic2dPseudoweightedMinic2d:
                    globalReportsExp2Minic2dPseudoweightedMinic2d = computeOutFileReports()
                return globalReportsExp2Minic2dPseudoweightedMinic2d
            else:
                raise raiseWrongWeightFormatException(weightFormat)
        else:
            raiseWrongBenchmarkFamException(benchmarkFam)
    elif counter == Counter.C2D:
        if benchmarkFam == BenchmarkFam.BAYES:
            if weightFormat == WeightFormat.DDNNF:
                global globalReportsExp2C2dBayesDdnnf
                if not globalReportsExp2C2dBayesDdnnf:
                    globalReportsExp2C2dBayesDdnnf = computeOutFileReports()
                return globalReportsExp2C2dBayesDdnnf
            else:
                raise raiseWrongWeightFormatException(weightFormat)
        elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
            if weightFormat == WeightFormat.DDNNF:
                global globalReportsExp2C2dPseudoweightedDdnnf
                if not globalReportsExp2C2dPseudoweightedDdnnf:
                    globalReportsExp2C2dPseudoweightedDdnnf = computeOutFileReports()
                return globalReportsExp2C2dPseudoweightedDdnnf
            else:
                raise raiseWrongWeightFormatException(weightFormat)
        else:
            raiseWrongBenchmarkFamException(benchmarkFam)
    elif counter == Counter.D4:
        if benchmarkFam == BenchmarkFam.BAYES:
            if weightFormat == WeightFormat.DDNNF:
                global globalReportsExp2D4BayesDdnnf
                if not globalReportsExp2D4BayesDdnnf:
                    globalReportsExp2D4BayesDdnnf = computeOutFileReports()
                return globalReportsExp2D4BayesDdnnf
            else:
                raiseWrongWeightFormatException(weightFormat)
        elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
            if weightFormat == WeightFormat.DDNNF:
                global globalReportsExp2D4PseudoweightedDdnnf
                if not globalReportsExp2D4PseudoweightedDdnnf:
                    globalReportsExp2D4PseudoweightedDdnnf = computeOutFileReports()
                return globalReportsExp2D4PseudoweightedDdnnf
            else:
                raiseWrongWeightFormatException(weightFormat)
        else:
            raiseWrongBenchmarkFamException(benchmarkFam)
    else:
        raiseWrongCounterException(counter)

################################################################################
'''Gets OUT file reports.'''

def getTriples(benchmarkFam, sortedByName=False): # [(counter, weightFormat, heuristicTuple=None),...]
    assert benchmarkFam in BenchmarkFam

    if benchmarkFam == BenchmarkFam.BAYES:
        triples = [
            (Counter.ADDMC, WeightFormat.MINIC2D, BEST_HEURISTIC_TUPLE),
            (Counter.D4, WeightFormat.DDNNF, None),
            (Counter.MINIC2D, WeightFormat.MINIC2D, None),
            (Counter.C2D, WeightFormat.DDNNF, None),
            (Counter.CACHET, WeightFormat.CACHET, None),
        ]
    if benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
        triples = [
            (Counter.D4, WeightFormat.DDNNF, None),
            (Counter.CACHET, WeightFormat.CACHET, None),
            (Counter.C2D, WeightFormat.DDNNF, None),
            (Counter.MINIC2D, WeightFormat.MINIC2D, None),
            (Counter.ADDMC, WeightFormat.MINIC2D, BEST_HEURISTIC_TUPLE),
        ]
    if benchmarkFam == BenchmarkFam.ALTOGETHER:
        triples = [
            (Counter.D4, WeightFormat.DDNNF, None),
            (Counter.C2D, WeightFormat.DDNNF, None),
            (Counter.MINIC2D, WeightFormat.MINIC2D, None),
            (Counter.ADDMC, WeightFormat.MINIC2D, BEST_HEURISTIC_TUPLE),
            (Counter.CACHET, WeightFormat.CACHET, None),
        ]

    if sortedByName:
        triples = sorted(triples, key=(lambda triple: triple[0].name))
    return triples

def getAllCountersOutFileReportsList(benchmarkFam, weightFormat, excludedCounters=set()):
    reportsList = []
    for (counter, weightFormat, heuristicTuple) in getTriples(benchmarkFam):
        if counter not in excludedCounters:
            reports = getOutFileReports(Experiment.COUNTERS, counter, benchmarkFam, weightFormat, heuristicTuple)
            reportsList.append(reports)
    return reportsList

def writeDataToCsvFile(csvFilePath, benchmarkFam, weightFormat):
    fields = [
        'counter',
        'benchmark',
        'status',
        'weightedModelCount',
        'seconds',
    ]
    with open(csvFilePath, 'w') as csvFile:
        csvFile.write('{}\n'.format(','.join(fields)))
        for outFileReport in mergeSublists(getAllCountersOutFileReportsList(benchmarkFam, weightFormat)):
            csvFile.write('{},'.format(outFileReport.counter))
            csvFile.write('{},'.format(outFileReport.cnfFileName))
            csvFile.write('{},'.format(outFileReport.outFileStatus))
            csvFile.write('{},'.format(outFileReport.modelCount))
            csvFile.write('{}\n'.format(outFileReport.time))

    print('\nOverwrote file: {}\n'.format(csvFilePath))

def getGlobalVarDefLines(globalVarName, benchmarkFam, counter=Counter.ADDMC, weightFormat=WeightFormat.MINIC2D):
    lines = ['{} = ['.format(globalVarName)]
    for outFileReport in getOutFileReports(Experiment.ALL_CONFIGS, counter, benchmarkFam, weightFormat):
        lines.append('\t{},'.format(repr(outFileReport)))
    lines.append(']\n')
    return lines

def writeDataToPyFile(pyFilePath):
    lines = [
        '#!/usr/bin/env python3\n',
        'from definitions import *\n',
        'from eval import Experiment, OutFileStatus, AddmcOutFileReport\n',
    ]

    global WRITING_DATA_TO_PY_FILE
    WRITING_DATA_TO_PY_FILE = True
    lines.extend(getGlobalVarDefLines('reportsExp1AddmcBayesMinic2d', BenchmarkFam.BAYES))
    lines.extend(getGlobalVarDefLines('reportsExp1AddmcPseudoweightedMinic2d', BenchmarkFam.PSEUDOWEIGHTED))
    WRITING_DATA_TO_PY_FILE = False

    lines = '\n'.join(lines)

    with open(pyFilePath, 'w') as pyFile:
        pyFile.write(lines)

    os.chmod(pyFilePath, stat.S_IRUSR ^ stat.S_IWUSR ^ stat.S_IXUSR) # owner reads/writes/executes

    print('\nOverwrote file: {}\n'.format(pyFilePath))

def mainDataCompression():
    # writeDataToCsvFile('data.csv', BenchmarkFam.ALTOGETHER, WeightFormat.WEIGHTED)
    writeDataToPyFile('data.py')

################################################################################
'''Analyzes correctness.'''

def isEqual(modelCount1, modelCount2):
    assert not math.isnan(modelCount1)
    assert not math.isnan(modelCount2)
    if {modelCount1, modelCount2} & {DUMMY_MODEL_COUNT} or math.isinf(modelCount1) or math.isinf(modelCount2):
        return False
    else:
        small = min(modelCount1, modelCount2)
        assert small >= 0.

        big = max(modelCount1, modelCount2)

        if small == 0. or big <= FLOAT_TOLERANCE_THRESHOLD:
            return big - small <= FLOAT_DIFF_TOLERANCE
        else:
            return big / small <= FLOAT_RATIO_TOLERANCE

def getCorrectnessAnalysisExp1(experiment, benchmarkFam, weightFormat):
    correctnessAnalysis = {} # cnfFileName |-> [(modelCount, heuristicTuple),...]
    for addmcOutFileReport in getOutFileReports(experiment, Counter.ADDMC, benchmarkFam, weightFormat):
        if addmcOutFileReport.isComplete():
            cnfFileName = addmcOutFileReport.cnfFileName
            modelCount = addmcOutFileReport.modelCount
            heuristicTuple = addmcOutFileReport.heuristicTuple
            if cnfFileName in correctnessAnalysis:
                (modelCount0, _) = correctnessAnalysis[cnfFileName][0]
                if not isEqual(modelCount0, modelCount):
                    correctnessAnalysis[cnfFileName].append((modelCount, heuristicTuple))
            else:
                correctnessAnalysis[cnfFileName] = [(modelCount, heuristicTuple)]

    diffDict = {}
    for cnfFileName in correctnessAnalysis:
        pairs = [(modelCount, getNamedHeuristicTuple(heuristicTuple)) for (modelCount, heuristicTuple) in correctnessAnalysis[cnfFileName]]
        if len(pairs) > 1:
            diffDict[cnfFileName] = pairs

    print('Correctness across heuristic configurations:', end='')
    if diffDict:
        print('\nNumber of benchmarks with different model counts: {}'.format(len(diffDict)))
        for cnfFileName in diffDict:
            print('\tBenchmark with different model counts: {}'.format(cnfFileName))
            for pair in diffDict[cnfFileName]:
                print('\t\t{}'.format(pair))
        print()
    else:
        print(' all good\n')

    return correctnessAnalysis

def getOutDirAnalysis(experiment, counter, weightFormat, benchmarkFam): # cnfFileName |-> (time, modelCount)
    outDirAnalysis = {}
    for outFileReport in getOutFileReports(experiment, counter, benchmarkFam, weightFormat):
        time = outFileReport.time
        modelCount = outFileReport.modelCount
        if outFileReport.isComplete():
            outDirAnalysis[outFileReport.cnfFileName] = (time, modelCount)

    return outDirAnalysis

def getCorrectnessAnalysisExp2(baseCounter, weightFormat, benchmarkFam): # cnfFileName |-> [(addmcOutFileReport, baseModelCount), ...]
    baseAnalysis = {}

    outDirAnalysis = getOutDirAnalysis(Experiment.COUNTERS, baseCounter, weightFormat, benchmarkFam)
    baseAnalysis.update(outDirAnalysis)
    correctnessAnalysis = {cnfFileName: [] for cnfFileName in baseAnalysis}

    def updateCorrectnessAnalysis(addmcOutFileReport):
        if addmcOutFileReport.isComplete():
            cnfFileName = addmcOutFileReport.cnfFileName
            if cnfFileName in correctnessAnalysis:
                (_, baseModelCount) = baseAnalysis[cnfFileName]
                if not isEqual(baseModelCount, addmcOutFileReport.modelCount):
                    correctnessAnalysis[cnfFileName].append(addmcOutFileReport)

    for addmcOutFileReport in getOutFileReports(Experiment.COUNTERS, Counter.ADDMC, benchmarkFam, WeightFormat.MINIC2D):
        updateCorrectnessAnalysis(addmcOutFileReport)

    print('Correctness vs counter {} with weight format {} on benchmark family {}:'.format(baseCounter.name, weightFormat.name, benchmarkFam.name), end='')
    diffDict = {cnfFileName: diffList for (cnfFileName, diffList) in correctnessAnalysis.items() if diffList}
    if diffDict:
        print('\nNumber of benchmarks with different model counts: {}'.format(len(diffDict)))
        verbose = 1
        if verbose:
            print('Model count differences:'.format(baseCounter.name))
            for cnfFileName in diffDict:
                print('\tBase model count: {}'.format(baseAnalysis[cnfFileName][1]))
                for addmcOutFileReport in diffDict[cnfFileName]:
                    print('\t\t{}'.format(addmcOutFileReport))
        print()
    else:
        print(' all good\n')

    return correctnessAnalysis

def analyzeCorrectness(experiment, benchmarkFam, weightFormat):
    print('Float equality tolerance: BIG - SMALL <= {} if SMALL = 0 or BIG <= {}, else BIG / SMALL <= {}\n'.format(FLOAT_DIFF_TOLERANCE, FLOAT_TOLERANCE_THRESHOLD, FLOAT_RATIO_TOLERANCE))
    if experiment.withConfigs():
        getCorrectnessAnalysisExp1(experiment, benchmarkFam, weightFormat)
    elif experiment == Experiment.COUNTERS:
        getCorrectnessAnalysisExp2(Counter.C2D, WeightFormat.DDNNF, benchmarkFam)
        getCorrectnessAnalysisExp2(Counter.CACHET, WeightFormat.CACHET, benchmarkFam)
        getCorrectnessAnalysisExp2(Counter.D4, WeightFormat.DDNNF, benchmarkFam)
        getCorrectnessAnalysisExp2(Counter.MINIC2D, WeightFormat.MINIC2D, benchmarkFam)
    else:
        raiseWrongExperimentException(experiment)

################################################################################
'''Analyzes performance.'''

def getHeuristicsPerformanceData(experiment, benchmarkFam): # heuristicTuple |-> [time,...] (possibly INF)
    performanceData = {}
    clusteringHeuristics = NO_LIN_CLUSTERING_CONFIGS if not INCLUDING_LINEAR else []
    outFileReports = getOutFileReports(experiment, Counter.ADDMC, benchmarkFam, WeightFormat.MINIC2D, heuristicTuple=None, clusteringHeuristics=clusteringHeuristics)
    for addmcOutFileReport in outFileReports:
        heuristicTuple = addmcOutFileReport.heuristicTuple
        time = addmcOutFileReport.time if addmcOutFileReport.isComplete() else INF # depends on INCLUDING_OVERFLOW_COMPLETIONS
        if heuristicTuple in performanceData:
            performanceData[heuristicTuple].append(time)
        else:
            performanceData[heuristicTuple] = [time]
    return performanceData

def getPerformanceAnalysisExp1(performanceData): # [(heuristicTuple, (completionCount, medianTime)),...]
    performanceAnalysis = {} # heuristicTuple |-> (completionCount, medianTime)
    for (heuristicTuple, times) in performanceData.items():
        completionCount = len([time for time in times if math.isfinite(time)])
        medianTime = statistics.median(times)
        performanceAnalysis[heuristicTuple] = (completionCount, medianTime)

    return sorted(performanceAnalysis.items(), key=(lambda kv: (-kv[1][0], kv[1][1]))) # sorted by big completionCount then by small medianTime

def getVbsTimes(outFileReportsList): # cnfFileName |-> bestTime (possibly INF)
    bestTimes = {}
    for outFileReports in outFileReportsList:
        for outFileReport in outFileReports:
            cnfFileName = outFileReport.cnfFileName
            time = outFileReport.time if outFileReport.isComplete() else INF
            if cnfFileName in bestTimes:
                bestTimes[cnfFileName] = min(bestTimes[cnfFileName], time)
            else:
                bestTimes[cnfFileName] = time
    return bestTimes

def getVbsAnalysis(outFileReportsList): # (completionCount, medianTime)
    d = getVbsTimes(outFileReportsList)
    completionCount = len([cnfFileName for cnfFileName in d if math.isfinite(d[cnfFileName])])
    medianTime = statistics.median(d.values())
    return (completionCount, medianTime)

def getUniqueBenchmarkCountsByHeuristicTuples(outFileReports): # heuristicTuple |-> uniqueBenchmarkCount
    heuristicTuples = {} # cnfFileName |-> {heuristicTuple,...}
    for outFileReport in outFileReports:
        if outFileReport.isComplete():
            cnfFileName = outFileReport.cnfFileName
            heuristicTuple = outFileReport.heuristicTuple
            if cnfFileName in heuristicTuples:
                heuristicTuples[cnfFileName].add(heuristicTuple)
            else:
                heuristicTuples[cnfFileName] = {heuristicTuple}

    uniqueBenchmarkCounts = collections.Counter()
    for heuristicTuples in heuristicTuples.values():
        if len(heuristicTuples) == 1:
            heuristicTuple = heuristicTuples.pop()
            uniqueBenchmarkCounts[heuristicTuple] += 1
    return uniqueBenchmarkCounts

def getFastestBenchmarkCountsByHeuristicTuples(outFileReports): # counter |-> fastestBenchmarkCount
    fastestTimes = {} # cnfFileName |-> time
    fastestHeuristicTuples = {} # cnfFileName |-> {heuristicTuple,...}
    for outFileReport in outFileReports:
        if outFileReport.isComplete():
            cnfFileName = outFileReport.cnfFileName
            time = outFileReport.time
            heuristicTuple = outFileReport.heuristicTuple
            if cnfFileName in fastestTimes:
                if fastestTimes[cnfFileName] == time:
                    fastestHeuristicTuples[cnfFileName].add(heuristicTuple)
                elif fastestTimes[cnfFileName] > time:
                    fastestTimes[cnfFileName] = time
                    fastestHeuristicTuples[cnfFileName] = {heuristicTuple}
            else:
                fastestTimes[cnfFileName] = time
                fastestHeuristicTuples[cnfFileName] = {heuristicTuple}

    fastestBenchmarkCounts = collections.Counter()
    for counters in fastestHeuristicTuples.values():
        if len(counters) == 1:
            heuristicTuple = counters.pop()
            fastestBenchmarkCounts[heuristicTuple] += 1
    return fastestBenchmarkCounts

def analyzePerformanceExp1(experiment, benchmarkFam, weightFormat):
    def printHeaderRow(smallWidth, bigWidth):
        for header in ['Row']:
            print(header.ljust(smallWidth), end='')
        for header in ['Clustering', 'Cluster var',]:
            print(header.ljust(bigWidth), end='')
        for header in ['Inv']:
            print(header.ljust(smallWidth), end='')
        for header in ['Diagram var']:
            print(header.ljust(bigWidth), end='')
        for header in ['Inv']:
            print(header.ljust(smallWidth), end='')
        for header in ['Unique', 'Fastest', 'Completions', 'Rate (%)', 'Med time (s)']:
            print(header.rjust(bigWidth), end='')
        print()

    def printVbsRow(smallWidth, bigWidth):
        print('0 '.rjust(smallWidth), end='')

        (completionCount, medianTime) = getVbsAnalysis([getOutFileReports(experiment, Counter.ADDMC, benchmarkFam, weightFormat)])

        print('VBS'.ljust(bigWidth), end='')
        print('VBS'.ljust(bigWidth), end='')
        print('VBS'.ljust(smallWidth), end='')
        print('VBS'.ljust(bigWidth), end='')
        print('VBS'.ljust(smallWidth), end='')
        print('NA'.rjust(bigWidth), end='')
        print('NA'.rjust(bigWidth), end='')

        print(str(completionCount).rjust(bigWidth), end='')
        print(getCompletionRateStr(benchmarkFam, completionCount, bigWidth), end='')
        print(FLOAT_FORMAT.format(medianTime).rjust(bigWidth))

    def printHeuristicTupleRows(smallWidth, bigWidth):
        def printConfigs(heuristicTuple):
            for namedHeuristic in getNamedHeuristicTuple(heuristicTuple):
                shortHeuristicName = getShortHeuristicName(namedHeuristic)
                if len(shortHeuristicName) == 1: # invClusterVar or invDiagramVar
                    print(shortHeuristicName.ljust(smallWidth), end='')
                else:
                    print(shortHeuristicName.ljust(bigWidth), end='')

        outFileReports = getOutFileReports(Experiment.ALL_CONFIGS, Counter.ADDMC, benchmarkFam, weightFormat)
        uniqueBenchmarkCounts = getUniqueBenchmarkCountsByHeuristicTuples(outFileReports)
        fastestBenchmarkCounts = getFastestBenchmarkCountsByHeuristicTuples(outFileReports)

        performanceAnalysis = getPerformanceAnalysisExp1(getHeuristicsPerformanceData(experiment, benchmarkFam))

        for i in range(len(performanceAnalysis)):
            print('{} '.format(i + 1).rjust(smallWidth), end='')

            (heuristicTuple, (completionCount, medianTime)) = performanceAnalysis[i]

            printConfigs(heuristicTuple)
            print(str(uniqueBenchmarkCounts[heuristicTuple]).rjust(bigWidth), end='')
            print(str(fastestBenchmarkCounts[heuristicTuple]).rjust(bigWidth), end='')
            print(str(completionCount).rjust(bigWidth), end='')
            print(getCompletionRateStr(benchmarkFam, completionCount, bigWidth), end='')
            print(FLOAT_FORMAT.format(medianTime).rjust(bigWidth))

    print('ADDMC experiment result on {} benchmarks in family {} with weight format {} and timeout {}:'.format(getBenchmarkCount(benchmarkFam), benchmarkFam.name, weightFormat.name, getTimeLimitStr(Experiment.ALL_CONFIGS)))

    smallWidth = 5
    bigWidth = 13

    printHeaderRow(smallWidth, bigWidth)
    printVbsRow(smallWidth, bigWidth)
    printHeuristicTupleRows(smallWidth, bigWidth)

def getCountersPerformanceData(benchmarkFam, weightFormat, heuristicTuple=None): # counter |-> [time,...] (possibly INF)
    performanceData = {}
    for counter in Counter:
        if counter == Counter.SHARPSAT and weightFormat.isWeighted():
            performanceData[counter] = [INF] * getBenchmarkCount(benchmarkFam)
        else:
            times = []
            for outFileReport in getOutFileReports(Experiment.COUNTERS, counter, benchmarkFam, weightFormat, heuristicTuple):
                time = outFileReport.time if outFileReport.isComplete() else INF # depends on INCLUDING_OVERFLOW_COMPLETIONS
                times.append(time)
            performanceData[counter] = times
    return performanceData

def getUniqueBenchmarkCountsByCounters(outFileReports): # counter |-> uniqueBenchmarkCount
    counters = {} # cnfFileName |-> {counter,...}
    for outFileReport in outFileReports:
        if outFileReport.isComplete():
            cnfFileName = outFileReport.cnfFileName
            counter = outFileReport.counter
            if cnfFileName in counters:
                counters[cnfFileName].add(counter)
            else:
                counters[cnfFileName] = {counter}

    uniqueBenchmarkCounts = collections.Counter()
    for counters in counters.values():
        if len(counters) == 1:
            counter = counters.pop()
            uniqueBenchmarkCounts[counter] += 1
    return uniqueBenchmarkCounts

def getFastestBenchmarkCountsByCounters(outFileReports): # counter |-> fastestBenchmarkCount
    fastestTimes = {} # cnfFileName |-> time
    fastestCounters = {} # cnfFileName |-> {counter,...}
    for outFileReport in outFileReports:
        if outFileReport.isComplete():
            cnfFileName = outFileReport.cnfFileName
            time = outFileReport.time
            counter = outFileReport.counter
            if cnfFileName in fastestTimes:
                if fastestTimes[cnfFileName] == time:
                    fastestCounters[cnfFileName].add(counter)
                elif fastestTimes[cnfFileName] > time:
                    fastestTimes[cnfFileName] = time
                    fastestCounters[cnfFileName] = {counter}
            else:
                fastestTimes[cnfFileName] = time
                fastestCounters[cnfFileName] = {counter}

    fastestBenchmarkCounts = collections.Counter()
    for counters in fastestCounters.values():
        if len(counters) == 1:
            counter = counters.pop()
            fastestBenchmarkCounts[counter] += 1
    return fastestBenchmarkCounts

def analyzePerformanceExp2(benchmarkFam, weightFormat):
    smallWidth = 14
    bigWidth = 25

    def printHeaderRow():
        print('Weight format'.ljust(smallWidth), end='')
        print('Counter'.ljust(bigWidth), end='')
        print('Unique'.rjust(smallWidth), end='')
        print('Fastest'.rjust(smallWidth), end='')
        print('Completions'.rjust(smallWidth), end='')
        print('Rate (%)'.rjust(smallWidth), end='')
        print('Med time (s)'.rjust(smallWidth))

    def printVbsRows(benchmarkFam, weightFormat):
        def printVbsRow(excludedCounters, vbsSuffix):
            (completionCount, medianTime) = getVbsAnalysis(getAllCountersOutFileReportsList(benchmarkFam, weightFormat, excludedCounters))

            print('{}'.format(weightFormat.name).ljust(smallWidth), end='')
            print('VBS_{}{}'.format(weightFormat.name, vbsSuffix).ljust(bigWidth), end='')
            print('NA'.rjust(smallWidth), end='')
            print('NA'.rjust(smallWidth), end='')
            print(str(completionCount).rjust(smallWidth), end='')
            print(getCompletionRateStr(benchmarkFam, completionCount, smallWidth), end='')
            print(FLOAT_FORMAT.format(medianTime).rjust(smallWidth))

        printVbsRow(set(), '')
        printVbsRow({Counter.ADDMC}, '-no_ADDMC')

    def printCounterRows(benchmarkFam, weightFormat):
        allCountersOutfileReports = mergeSublists(getAllCountersOutFileReportsList(benchmarkFam, weightFormat))
        uniqueBenchmarkCounts = getUniqueBenchmarkCountsByCounters(allCountersOutfileReports) # counter |-> uniqueBenchmarkCount
        fastestBenchmarkCounts = getFastestBenchmarkCountsByCounters(allCountersOutfileReports) # counter |-> fastestBenchmarkCount

        def printCounterRow(counter, weightFormat, heuristicTuple):
            completionCount = 0
            times = []
            outFileReports = getOutFileReports(Experiment.COUNTERS, counter, benchmarkFam, weightFormat, heuristicTuple)
            for outFileReport in outFileReports:
                if outFileReport.isComplete():
                    completionCount += 1
                times.append(outFileReport.time)

            medianTime = statistics.median(times)

            print('{}'.format(weightFormat.name).ljust(smallWidth), end='')
            print('{}'.format(counter.name).ljust(bigWidth), end='')
            print(str(uniqueBenchmarkCounts[counter]).rjust(smallWidth), end='')
            print(str(fastestBenchmarkCounts[counter]).rjust(smallWidth), end='')
            print(str(completionCount).rjust(smallWidth), end='')
            print(getCompletionRateStr(benchmarkFam, completionCount, smallWidth), end='')
            print(FLOAT_FORMAT.format(medianTime).rjust(smallWidth))

        for triple in getTriples(benchmarkFam):
            printCounterRow(*triple)

    print('Performance summary {{benchmark family: {} ({}), weight format: {}}}:'.format(benchmarkFam.name, getBenchmarkCount(benchmarkFam), weightFormat.name))

    printHeaderRow()
    printVbsRows(benchmarkFam, weightFormat)
    printCounterRows(benchmarkFam, weightFormat)

def analyzePerformance(experiment, benchmarkFam, weightFormat):
    if experiment.withConfigs():
        analyzePerformanceExp1(experiment, benchmarkFam, weightFormat)
    elif experiment == Experiment.COUNTERS:
        analyzePerformanceExp2(benchmarkFam, weightFormat)
    else:
        raiseWrongExperimentException(experiment)

################################################################################
'''Analyzes OUT file reports.'''

def getAddmcErrors(experiment, benchmarkFam, weightFormat):
    if weightFormat.isWeighted():
        weightFormat = WeightFormat.getSpecificWeightedFormat(Counter.ADDMC)

    errors = set()
    for addmcOutFileReport in getOutFileReports(experiment, Counter.ADDMC, benchmarkFam, weightFormat):
        if addmcOutFileReport.outFileStatus == OutFileStatus.ERROR:
            errors.add(addmcOutFileReport.cnfFileName)

    print('ADDMC errors for benchmark family {} with weight format {}: {}'.format(benchmarkFam.name, weightFormat.name, format(len(errors))))
    if errors:
        print(sorted(errors))
    print()

def writeAnalysis(experiment):
    def analyzeIndividuals(benchmarkFams):
        for (benchmarkFam, weightFormat) in benchmarkFams:
            print()
            printSeparator(':')
            print('\nAnalyzing benchmark family {} with weight format {}\n'.format(benchmarkFam.name, weightFormat.name))

            getAddmcErrors(experiment, benchmarkFam, weightFormat)

            if INCLUDING_OVERFLOW_COMPLETIONS:
                print('Including overflow completions\n')
            else:
                print('Excluding overflow completions\n')

            analyzeCorrectness(experiment, benchmarkFam, weightFormat)
            analyzePerformance(experiment, benchmarkFam, weightFormat)

    def analyzeWhole():
        print()
        printSeparator(':')
        print('\nAnalyzing all benchmark families\n')
        analyzePerformance(experiment, BenchmarkFam.ALTOGETHER, WeightFormat.WEIGHTED)

    benchmarkFams = [
        (BenchmarkFam.BAYES, WeightFormat.WEIGHTED),
        (BenchmarkFam.PSEUDOWEIGHTED, WeightFormat.WEIGHTED),
    ]

    if experiment == Experiment.ALL_CONFIGS:
        txtFileName = 'exp1lin1.txt' if INCLUDING_LINEAR else 'exp1lin0.txt'
    elif experiment == Experiment.BEST_CONFIGS:
        txtFileName = 'exp1b.txt'
    elif experiment == Experiment.COUNTERS:
        txtFileName = 'exp2.txt'
    else:
        raiseWrongExperimentException(experiment)

    txtFilePath = os.path.join(ANALYSIS_PATH, 'tables', txtFileName)
    print('Overwriting file: {}'.format(txtFilePath))

    stdOut = sys.stdout
    with open(txtFilePath, 'w') as txtFile:
        sys.stdout = txtFile
        analyzeIndividuals(benchmarkFams)
        analyzeWhole()
        sys.stdout = stdOut

def mainAnalysis():
    def writeAnalysisExp1():
        global INCLUDING_LINEAR
        INCLUDING_LINEAR = True
        writeAnalysis(Experiment.ALL_CONFIGS)

    def writeAnalysisExp1NoLin():
        global INCLUDING_LINEAR
        INCLUDING_LINEAR = False
        writeAnalysis(Experiment.ALL_CONFIGS)

    def writeAnalysisExp1B():
        writeAnalysis(Experiment.BEST_CONFIGS)

    def writeAnalysisExp2():
        writeAnalysis(Experiment.COUNTERS)

    print()
    writeAnalysisExp1()
    writeAnalysisExp1NoLin()
    writeAnalysisExp1B()
    writeAnalysisExp2()
    print()

################################################################################
'''Handles plots.'''

LOG_SCALE_MIN = 1e-3

CLUSTERING_HEURISTIC_SHORT_NAMES = {
    ClusteringHeuristic.MONOLITHIC: 'Mono',
    ClusteringHeuristic.LINEAR: 'Linear',
    ClusteringHeuristic.BUCKET_LIST: 'BE-List',
    ClusteringHeuristic.BUCKET_TREE: 'BE-Tree',
    ClusteringHeuristic.BOUQUET_LIST: 'BM-List',
    ClusteringHeuristic.BOUQUET_TREE: 'BM-Tree'
}

def getHeuristicCurveName(clusteringHeuristic):
    return CLUSTERING_HEURISTIC_SHORT_NAMES[clusteringHeuristic]

def getCounterCurveName(counter, heuristicTuple):
    name = counter.value
    if heuristicTuple == BEST_HEURISTIC_TUPLE:
        pass
    elif heuristicTuple != None:
        name = CLUSTERING_HEURISTIC_SHORT_NAMES[ClusteringHeuristic(heuristicTuple[0])]
    return name

def getTimes(outFileReports): # [time,...]
    times = []
    for outFileReport in outFileReports:
        if outFileReport.isComplete():
            times.append(outFileReport.time)
    return times

def getCurveFormat(curveIndex):
    STYLES = [
        '--',
        '-',
        ':',
        '-.',
        # '*',
        # 'v',
    ]
    COLORS = [
        'b', # blue
        'g', # green
        'r', # red
        'c', # cyan
        'm', # magenta
        'y', # yellow
        'k', # black
    ]
    return STYLES[curveIndex % len(STYLES)] + COLORS[curveIndex % len(COLORS)]

def drawCurve(times, axes, curveName, curveIndex):
    def getPoints(times): # ([x_0,...], [y_0,...])
        ys = sorted(times)
        xs = range(1, len(ys) + 1)
        # return (ys, xs)
        return (xs, ys)

    (xs, ys) = getPoints(times)
    axes.plot(xs, ys, getCurveFormat(curveIndex), markevery=30, label=curveName)

def plotConfigs(axes, benchmarkFam, weightFormat, experiment):
    def getAddmcOutFileReports(heuristicTuple):
        return getOutFileReports(experiment, Counter.ADDMC, benchmarkFam, weightFormat, heuristicTuple)

    def getAddmcOutFileReportsList(heuristicTuples):
        return [getAddmcOutFileReports(heuristicTuple) for heuristicTuple in heuristicTuples]

    nonlocalCurveIndex = -1 # will be incremented before drawing

    def drawHeuristicVbsCurve(curveName):
        nonlocal nonlocalCurveIndex
        nonlocalCurveIndex += 1

        heuristicTuples = getHeuristicTuples(experiment)
        outFileReportsList = getAddmcOutFileReportsList(heuristicTuples)
        drawCurve(getVbsTimes(outFileReportsList).values(), axes, curveName, nonlocalCurveIndex)

    def drawHeuristicCurve(heuristicTuple, curveName):
        nonlocal nonlocalCurveIndex
        nonlocalCurveIndex += 1

        drawCurve(getTimes(getAddmcOutFileReports(heuristicTuple)), axes, curveName, nonlocalCurveIndex)

    if experiment == Experiment.ALL_CONFIGS:
        # drawHeuristicVbsCurve('VBS245')
        drawHeuristicCurve(BEST_HEURISTIC_TUPLE, 'Best1')
        drawHeuristicCurve(BEST_HEURISTIC_TUPLE_2, 'Best2')
        drawHeuristicCurve(NO_LIN_MEDIAN_HEURISTIC_TUPLE, 'Median')
        # drawHeuristicCurve(BEST_HEURISTIC_TUPLE_MONO, 'Best-Mono')
        drawHeuristicCurve(WORST_HEURISTIC_TUPLE, 'Worst')
    elif experiment == Experiment.BEST_CONFIGS:
        drawHeuristicVbsCurve('VBS5')
        drawHeuristicCurve(BEST_HEURISTIC_TUPLE, 'Best1')
        drawHeuristicCurve(BEST_HEURISTIC_TUPLE_2, 'Best2')
        drawHeuristicCurve(BEST_HEURISTIC_TUPLE_3, 'Best3')
        drawHeuristicCurve(BEST_HEURISTIC_TUPLE_4, 'Best4')
        drawHeuristicCurve(BEST_HEURISTIC_TUPLE_5, 'Best5')
    else:
        raiseWrongExperimentException(experiment)

    axes.legend(loc='lower right')

def plotCounters(axes, benchmarkFam, weightFormat):
    nonlocalCurveIndex = -1 # will be incremented before drawing

    def drawCounterVbsCurve(curveName, excludedCounters):
        nonlocal nonlocalCurveIndex
        nonlocalCurveIndex += 1

        outFileReportsList = getAllCountersOutFileReportsList(benchmarkFam, weightFormat, excludedCounters)
        drawCurve(getVbsTimes(outFileReportsList).values(), axes, curveName, nonlocalCurveIndex)

    def drawCounterCurve(counter, weightFormat, heuristicTuple=None):
        nonlocal nonlocalCurveIndex
        nonlocalCurveIndex += 1

        outFileReports = getOutFileReports(Experiment.COUNTERS, counter, benchmarkFam, weightFormat, heuristicTuple)
        drawCurve(getTimes(outFileReports), axes, getCounterCurveName(counter, heuristicTuple), nonlocalCurveIndex)

    def adjustLegend(): # [a, b, c,..., z] |-> [z, a, b, c,...]
        (handles, labels) = axes.get_legend_handles_labels()

        handles = handles[-1:] + handles[:-1]
        labels = labels[-1:] + labels[:-1]

        axes.legend(handles, labels)

    drawCounterVbsCurve('VBS1', set())
    drawCounterVbsCurve('VBS0', {Counter.ADDMC})

    for (counter, weightFormat, heuristicTuple) in getTriples(benchmarkFam, sortedByName=False):
        drawCounterCurve(counter, weightFormat, heuristicTuple)

    # adjustLegend()

    axes.legend(
        loc='lower right'
    )

def getFigSize(figSize): # AAAI format
    if not figSize:
        columnWidthPt = 239.39438 # TeX \the\columnwidth
        figWidthPt = columnWidthPt * .95 # AAAI style
        figWidthIn = figWidthPt / 72
        figWidthIn *= 2 # scale
        figHeightIn = figWidthIn * .7 # ratio
        figSize = (figWidthIn, figHeightIn)
    return figSize

def getFigAndAxes(figSize=None): # figSize=(figWidthIn, figHeightIn)
    plt.rcParams.update({
        'text.usetex': True, # type 1 font
        'font.size': 13,
    })

    (fig, axes) = plt.subplots(figsize=getFigSize(figSize))

    axes.grid(linewidth='.4')

    return (fig, axes)

def saveFig(fig, figBaseName):
    for ext in ['png', 'pdf']:
        figPath = os.path.join(FIGURES_PATH, '{}.{}'.format(figBaseName, ext))
        fig.savefig(figPath, bbox_inches='tight')
        print('Ovewrote file: {}'.format(figPath))

def plotFig(experiment, benchmarkFam, weightFormat):
    def getBenchmarkFamCode(benchmarkFam):
        if benchmarkFam == BenchmarkFam.BAYES:
            return 'bayes'
        elif benchmarkFam == BenchmarkFam.PSEUDOWEIGHTED:
            return 'pseudoweighted'
        elif benchmarkFam == BenchmarkFam.ALTOGETHER:
            return 'altogether'
        else:
            raiseWrongBenchmarkFamException(benchmarkFam)

    (fig, axes) = getFigAndAxes()

    if experiment.withConfigs():
        plotConfigs(axes, benchmarkFam, weightFormat, experiment)
    elif experiment == Experiment.COUNTERS:
        plotCounters(axes, benchmarkFam, weightFormat)
    else:
        raiseWrongExperimentException(experiment)

    axes.set_yscale('log')

    axes.set_xlim(0, 2000)
    axes.set_ylim(LOG_SCALE_MIN, getTimeLimit(experiment))

    axes.set_xlabel('Number of benchmarks solved')
    axes.set_ylabel('Longest solving time (seconds)')

    # axes.set_title('Experiment {}'.format(experiment.value))

    # plt.axvline(x=10, color='gray', linewidth=2, linestyle='-')
    # plt.axhline(y=10, color='gray', linewidth=2, linestyle='-')

    saveFig(fig, 'fig-exp{}-{}'.format(experiment.value, getBenchmarkFamCode(benchmarkFam)))

def mainPlotting():
    print('\nMaking figures...')

    plotFig(Experiment.ALL_CONFIGS, BenchmarkFam.ALTOGETHER, WeightFormat.MINIC2D)
    plotFig(Experiment.ALL_CONFIGS, BenchmarkFam.BAYES, WeightFormat.MINIC2D)
    plotFig(Experiment.ALL_CONFIGS, BenchmarkFam.PSEUDOWEIGHTED, WeightFormat.MINIC2D)

    plotFig(Experiment.BEST_CONFIGS, BenchmarkFam.ALTOGETHER, WeightFormat.MINIC2D)
    plotFig(Experiment.BEST_CONFIGS, BenchmarkFam.BAYES, WeightFormat.MINIC2D)
    plotFig(Experiment.BEST_CONFIGS, BenchmarkFam.PSEUDOWEIGHTED, WeightFormat.MINIC2D)

    plotFig(Experiment.COUNTERS, BenchmarkFam.ALTOGETHER, WeightFormat.WEIGHTED)
    plotFig(Experiment.COUNTERS, BenchmarkFam.BAYES, WeightFormat.WEIGHTED)
    plotFig(Experiment.COUNTERS, BenchmarkFam.PSEUDOWEIGHTED, WeightFormat.WEIGHTED)

    print('Made figures.\n')

################################################################################
'''Notes.'''

def noteBenchmarkVarsClauses():
    varCounts = {} # cnfFileName |-> varCount
    clauseCounts = {} # cnfFileName |-> clauseCount
    for cnfFilePath in getCnfFilePaths(BenchmarkFam.ALTOGETHER, WeightFormat.MINIC2D, Counter.ADDMC):
        for line in open(cnfFilePath):
            if line.startswith('p cnf'):
                (varCount, clauseCount) = line.split()[2:]
                cnfFileName = getFileName(cnfFilePath)
                varCounts[cnfFileName] = int(varCount)
                clauseCounts[cnfFileName] = int(clauseCount)
                break

    varClausePairs = [(varCounts[cnfFileName], clauseCounts[cnfFileName]) for cnfFileName in varCounts]

    clausesOverVarsPairs = [varClausePair[1] / varClausePair[0] for varClausePair in varClausePairs]
    minClausesOverVars = '{:.1f}'.format(min(clausesOverVarsPairs))
    medianClausesOverVars = '{:.1f}'.format(statistics.median(clausesOverVarsPairs))
    maxClausesOverVars = '{:.1f}'.format(max(clausesOverVarsPairs))

    vars = [varClausePair[0] for varClausePair in varClausePairs]
    maxVars = format(max(vars), ',')

    clauses = [varClausePair[1] for varClausePair in varClausePairs]
    medianClauses = format(statistics.median(clauses), ',')
    maxClauses = format(max(clauses), ',')

    def plotVarsClauses():
        (fig, axes) = getFigAndAxes((11, 5))
        axes.boxplot((vars, clauses, clausesOverVarsPairs), showfliers=False)

        axes.set_yscale('log')

        # axes.set_xlabel('\\#X (min \\#X \\& median \\#X \\& max \\#X)')
        axes.set_xticklabels([
            '\\#vars ({} \\& {} \\& {})'.format(min(vars), statistics.median(vars), maxVars),
            '\\#clauses ({} \\& {} \\& {})'.format(min(clauses), medianClauses, maxClauses),
            '\\#clauses / \\#vars ({} \\& {} \\& {})'.format(minClausesOverVars, medianClausesOverVars, maxClausesOverVars),
        ])

        saveFig(fig, 'vars-clauses')

    print()
    plotVarsClauses()

def noteMavcs():
    MIN_MAVC = 4
    MAX_SOLVED_MAVC = 246
    MAX_MAVC = 7663

    def getOutFileReportDict(): # cnfFileName |-> outFileReport
        outFileReportDict = {}
        for outFileReport in getOutFileReports(Experiment.COUNTERS, Counter.ADDMC, BenchmarkFam.ALTOGETHER, WeightFormat.MINIC2D):
            outFileReportDict[outFileReport.cnfFileName] = outFileReport
        return outFileReportDict

    def getMavcsAndOutFileReports(): # mavc |-> [outFileReport,...]
        outFileReportDict = getOutFileReportDict()
        mavcsAndOutFileReports = {}
        for mavcOutFilePath in getFilePaths(MAVC_DATA_PATH, '.out', emptyOk=False):
            mavcOutFileReport = getAddmcOutFileReport(mavcOutFilePath, Experiment.COUNTERS)
            mavc = mavcOutFileReport.mavc
            if mavc != None:
                cnfFileName = mavcOutFileReport.cnfFileName
                outFileReport = outFileReportDict[cnfFileName]
                if mavc in mavcsAndOutFileReports:
                    mavcsAndOutFileReports[mavc].append(outFileReport)
                else:
                    mavcsAndOutFileReports[mavc] = [outFileReport]
        return mavcsAndOutFileReports

    def getMavcsAndOutFileReportsSplitByBenchmarkFams(): # (bayesMavcsAndOutFileReports, pseudoweightedMavcsAndOutFileReports)
        mavcsAndOutFileReports = getMavcsAndOutFileReports()

        bayesMavcsAndOutFileReports = {mavc: [] for mavc in mavcsAndOutFileReports}
        pseudoweightedMavcsAndOutFileReports = {mavc: [] for mavc in mavcsAndOutFileReports}

        for (mavc, outFileReports) in mavcsAndOutFileReports.items():
            for outFileReport in outFileReports:
                if isBayesBenchmark(outFileReport.cnfFileName):
                    bayesMavcsAndOutFileReports[mavc].append(outFileReport)
                else:
                    pseudoweightedMavcsAndOutFileReports[mavc].append(outFileReport)

        return (bayesMavcsAndOutFileReports, pseudoweightedMavcsAndOutFileReports)

    def getMavcsAndBenchmarkCounts(benchmarkFam, incompleteOk, timeLimit):
        assert timeLimit == INF or not incompleteOk

        if benchmarkFam == BenchmarkFam.ALTOGETHER:
            mavcsAndOutFileReports = getMavcsAndOutFileReports()
        else:
            (bayesMavcsAndOutFileReports, pseudoweightedMavcsAndOutFileReports) = getMavcsAndOutFileReportsSplitByBenchmarkFams()
            mavcsAndOutFileReports = bayesMavcsAndOutFileReports if benchmarkFam == BenchmarkFam.BAYES else pseudoweightedMavcsAndOutFileReports

        mavcsAndBenchmarkCounts = {}
        for (mavc, outFileReports) in mavcsAndOutFileReports.items():
            benchmarkCount = 0
            for outFileReport in outFileReports:
                if incompleteOk or outFileReport.isComplete():
                    if outFileReport.time <= timeLimit:
                        benchmarkCount += 1
            mavcsAndBenchmarkCounts[mavc] = benchmarkCount
        return mavcsAndBenchmarkCounts

    def getCactusXsYs(benchmarkFam, incompleteOk=True, timeLimit=INF): # (xs, ys) where each x = MAVC and y = benchmarkCount
        mavcsAndBenchmarkCounts = getMavcsAndBenchmarkCounts(benchmarkFam, incompleteOk, timeLimit)
        mavcsAndBenchmarkCounts = sorted(mavcsAndBenchmarkCounts.items())

        (mavc, benchmarkCount) = mavcsAndBenchmarkCounts[0]
        xs = [mavc]
        ys = [benchmarkCount]

        for (mavc, benchmarkCount) in mavcsAndBenchmarkCounts:
            if benchmarkCount > 0:
                xs.append(mavc)
                ys.append(benchmarkCount + ys[-1])

        return (xs, ys)

    def plotBenchmarkCounts():
        (altogetherXs, altogetherYs) = getCactusXsYs(BenchmarkFam.ALTOGETHER)
        (altogetherSolvedXs, altogetherSolvedYs) = getCactusXsYs(BenchmarkFam.ALTOGETHER, incompleteOk=False)

        (fig, axes) = getFigAndAxes()

        curveIndex = 0

        axes.plot(altogetherXs, altogetherYs, getCurveFormat(curveIndex), label='Benchmarks with known MAVCs'),
        curveIndex += 1
        axes.plot(altogetherSolvedXs, altogetherSolvedYs, getCurveFormat(curveIndex), label='Benchmarks solved by ADDMC')

        axes.set_xscale('log')

        axes.set_xlim(1, 1e4) # MIN_MAVC == 4, MAX_MAVC = 7663
        axes.set_ylim(0, 2000) # 1906 known MAVCs

        axes.set_xlabel('Upper bound for MAVC (maximum ADD variable count)')
        axes.set_ylabel('Number of benchmarks')

        axes.legend(loc='lower right')

        saveFig(fig, 'fig-MAVC-benchmarks')

    def getBenchmarkCountsAndCompletionTimes(outFileReportDict): # [(mavc, benchmarkCount, [completionTime,...]),...]
        benchmarkCounts = collections.Counter() # mavc |-> benchmarkCount
        completionTimes = {} # mavc |-> [completionTime,...]

        for outFilePath in getFilePaths(MAVC_DATA_PATH, '.out', emptyOk=False):
            outFileReport = getAddmcOutFileReport(outFilePath, Experiment.COUNTERS)
            cnfFileName = outFileReport.cnfFileName
            mavc = outFileReport.mavc
            if mavc != None:
                benchmarkCounts[mavc] += 1
                if mavc not in completionTimes:
                    completionTimes[mavc] = []
                if outFileReportDict[outFileReport.cnfFileName].isComplete():
                    completionTimes[mavc].append(outFileReport.time)

        benchmarkCountsAndCompletionTimes = [
            (mavc, benchmarkCounts[mavc], completionTimes[mavc])
            for mavc in sorted(benchmarkCounts)
        ]

        return benchmarkCountsAndCompletionTimes

    def plotTimes():
        benchmarkCountsAndCompletionTimes = getBenchmarkCountsAndCompletionTimes(getOutFileReportDict())

        mavcs = []
        times = []
        for (mavc, benchmarkCount, completionTimes) in benchmarkCountsAndCompletionTimes:
            for completionTime in completionTimes:
                mavcs.append(mavc)
                times.append(completionTime)

        (fig, axes) = getFigAndAxes()
        axes.scatter(mavcs, times)

        # axes.set_xscale('log')
        axes.set_yscale('log')

        axes.set_xlim(0, 250)
        axes.set_ylim(LOG_SCALE_MIN, getTimeLimit(Experiment.COUNTERS))

        axes.set_xlabel('MAVC (maximum ADD variable count)')
        axes.set_ylabel('ADDMC solving time (seconds)')

        saveFig(fig, 'fig-MAVC-times')

    def plotCompletionRates():
        benchmarkCountsAndCompletionTimes = getBenchmarkCountsAndCompletionTimes(getOutFileReportDict())

        mavcs = []
        completionRates = []
        for (mavc, benchmarkCount, completionTimes) in benchmarkCountsAndCompletionTimes:
            mavcs.append(mavc)
            completionRates.append(len(completionTimes) / benchmarkCount * 100)

        (fig, axes) = getFigAndAxes()
        axes.scatter(mavcs, completionRates)

        axes.set_xscale('log')

        axes.set_xlabel('Maximum ADD variable count')
        axes.set_ylabel('Percentage of benchmarks solved')

        saveFig(fig, 'fig-MAVC-rates')

    def groupBenchmarkCountsAndCompletionTimes(separators=[17, 24, 32, 39, 46, 62, 72, 112, 187, 247, INF]): # [([mavc,...], benchmarkCount, [completionTime,...]),...]
        benchmarkCountsAndCompletionTimes = getBenchmarkCountsAndCompletionTimes(getOutFileReportDict())

        groups = []
        groupMavcs = []
        groupBenchmarkCount = 0
        groupCompletionTimes = []
        group = 0
        for (mavc, benchmarkCount, completionTimes) in benchmarkCountsAndCompletionTimes:
            seperator = separators[group]
            if mavc < seperator:
                groupMavcs.append(mavc)
                groupBenchmarkCount += benchmarkCount
                groupCompletionTimes.extend(completionTimes)
            else:
                groups.append((groupMavcs, groupBenchmarkCount, groupCompletionTimes))
                groupMavcs = [mavc]
                groupBenchmarkCount = benchmarkCount
                groupCompletionTimes = completionTimes.copy()
                group += 1
        groups.append((groupMavcs, groupBenchmarkCount, groupCompletionTimes))

        width = 11
        floatFormat = '{{:{}.2f}}'.format(width)
        midLine = ' & '
        endLine = r' \\'


        headers = (
            'Low MAVC',
            'High MAVC',
            'MAVC count',
            'B count',
            'Solved',
            'Percent',
            'Median time', # all benchmarks
            # 'Mean time', # only solved benchmarks
        )
        headerCount = len(headers)
        for i in range(headerCount):
            print(headers[i].ljust(width), end=(endLine if i == headerCount - 1 else midLine))
        print()

        for (mavcs, benchmarkCount, completionTimes) in groups:
            completions = len(completionTimes)
            entries = (
                mavcs[0],
                mavcs[-1],
                len(mavcs),
                benchmarkCount,
                completions,
                completions / benchmarkCount * 100,
                statistics.median(completionTimes + [INF] * (benchmarkCount - completions)) if completionTimes else INF,
                # statistics.mean(completionTimes) if completionTimes else INF,
            )
            entryCount = len(entries)
            for i in range(entryCount):
                num = entries[i]
                if type(num) == int:
                    num = str(num).rjust(width)
                elif math.isinf(num):
                    num = r'\infty'.rjust(width)
                else:
                    num = floatFormat.format(num)
                print(num, end=(endLine if i == entryCount - 1 else midLine))
            print()

        return groups

    print()

    plotBenchmarkCounts()
    plotTimes()
    # plotCompletionRates()

    print()
    groupBenchmarkCountsAndCompletionTimes([247, INF])
    print()
    groupBenchmarkCountsAndCompletionTimes([70, 101, INF])

def noteOverflows():
    def noteOverflow(counter, weightFormat):
        overflowCount = 0
        for outFileReport in getOutFileReports(Experiment.COUNTERS, counter, BenchmarkFam.ALTOGETHER, weightFormat):
            modelCount = outFileReport.modelCount
            if modelCount != None and math.isinf(modelCount):
                overflowCount += 1
        print('Overflows: {}, of {} -- {} & {}'.format(str(overflowCount).rjust(2), getBenchmarkCount(BenchmarkFam.ALTOGETHER), counter, weightFormat))

    print()
    noteOverflow(Counter.ADDMC, WeightFormat.MINIC2D)
    noteOverflow(Counter.C2D, WeightFormat.DDNNF)
    noteOverflow(Counter.CACHET, WeightFormat.CACHET)
    noteOverflow(Counter.D4, WeightFormat.DDNNF)
    noteOverflow(Counter.MINIC2D, WeightFormat.MINIC2D)

def noteReasonerTime():
    compilerTimes = {} # cnfFileName |-> time
    totalTimes = {} # cnfFileName |-> time
    for outFileReport in getOutFileReports(Experiment.COUNTERS, Counter.C2D, BenchmarkFam.ALTOGETHER, WeightFormat.DDNNF):
        time = outFileReport.time
        cnfFileName = outFileReport.cnfFileName
        if math.isfinite(time):
            compilerTimes[cnfFileName] = outFileReport.compilerTime
            totalTimes[cnfFileName] = time
    compilerTimeRate = sum(compilerTimes.values()) / sum(totalTimes.values()) * 100
    print('\nBayes weighted counting: c2d compiler\'s average portion of total time:{}%'.format(FLOAT_FORMAT.format(compilerTimeRate)))

def noteModelCountDiffs(benchmarkFam, weightFormat):
    allCounters = Counter.getWeightedCounters()
    numCounters = len(allCounters)

    benchmarks = {} # cnfFileName |-> {Counter: modelCount,...}
    for outFileReport in mergeSublists(getAllCountersOutFileReportsList(benchmarkFam, weightFormat)):
        if outFileReport.isComplete():
            cnfFileName = outFileReport.cnfFileName
            counter = outFileReport.counter
            modelCount = outFileReport.modelCount
            if cnfFileName in benchmarks:
                benchmarks[cnfFileName][counter] = modelCount
            else:
                benchmarks[cnfFileName] = {counter: modelCount}

    benchmarks = {
        cnfFileName: countersModelCounts for (cnfFileName, countersModelCounts) in benchmarks.items()
        if len(countersModelCounts) == numCounters
    }
    print('\nBenchmarks solved by all {} counters: {}'.format(numCounters, len(benchmarks)))

    def isDiffFromAllOthers(baseCounter, countersModelCounts): # countersModelCounts: Counter |-> modelCount
        assert len(countersModelCounts) == numCounters
        for counter in countersModelCounts:
            if counter != baseCounter:
                if isEqual(countersModelCounts[counter], countersModelCounts[baseCounter]):
                    return False
        return True

    allDiffs = collections.Counter()
    for cnfFileName in benchmarks:
        countersModelCounts = benchmarks[cnfFileName]
        for counter in allCounters:
            if isDiffFromAllOthers(counter, countersModelCounts):
                allDiffs[counter] += 1

    print('Benchmarks whose a counter\'s answers differ from all other counters\' answers:')
    print(sorted(allDiffs.items(), key=(lambda kv: kv[0].name)))

def noteCachetDummyModelCounts():
    print()
    getOutFileReports(Experiment.COUNTERS, Counter.CACHET, BenchmarkFam.ALTOGETHER, WeightFormat.CACHET) # populates globalCachetDummyModelCounts1
    dummiesFilePath = os.path.join(ANALYSIS_PATH, 'texts', 'dummies.txt')
    with open(dummiesFilePath, 'w') as dummiesFile:
        dummiesFile.write('len(globalCachetDummyModelCounts1) == {}\n'.format(len(globalCachetDummyModelCounts1)))
        dummiesFile.write('globalCachetDummyModelCounts1 == {}\n\n'.format(globalCachetDummyModelCounts1))
        dummiesFile.write('len(globalCachetDummyModelCounts2) == {}\n'.format(len(globalCachetDummyModelCounts2)))
        dummiesFile.write('globalCachetDummyModelCounts2 == {}\n'.format(globalCachetDummyModelCounts2))
    print('Overwrote file {}'.format(dummiesFilePath))

def noteBayesModelCountDistribution():
    print()
    modelCounts = []
    for outFileReport in getOutFileReports(Experiment.COUNTERS, Counter.ADDMC, BenchmarkFam.BAYES, WeightFormat.MINIC2D, BEST_HEURISTIC_TUPLE):
        modelCounts.append(outFileReport.modelCount)

    for lowerBound in [
        1e-3,
        1e-5,
        1e-6,
    ]:
        lowModelCounts = [modelCount for modelCount in modelCounts if modelCount != None and 0. <= modelCount <= lowerBound]
        print('Upper bound: {}'.format(lowerBound))
        print('\tSmall non-negative model counts: {}'.format(len(lowModelCounts)))
        print('\tSmall non-negative model count rate: {}\n'.format(FLOAT_FORMAT.format(len(lowModelCounts) / len(modelCounts) * 100)))

def noteMissingSlurmOutputs(outDir):
    print('OUT dir: {}'.format(outDir))
    outDir = os.path.abspath(outDir)

    def getExistingRandNum():
        return int(outDir.split('-')[-1])

    def getExistingScriptsPath():
        return os.path.join(outDir, 'AutoScripts-{}'.format(getExistingRandNum()))

    def getExistingCountingScriptsPath():
        return os.path.join(getExistingScriptsPath(), 'counting')

    def getExistingSubmittingScriptsPath():
        return os.path.join(getExistingScriptsPath(), 'submitting')

    def getExistingSlurmOutputsPath():
        return os.path.join(outDir, 'AutoOutputs-{}'.format(getExistingRandNum()), 'slurm')

    def getCountingScriptCount():
        return len(getFilePaths(getExistingCountingScriptsPath(), '.sh'))

    def getMissingOutputIndicies():
        outFileIndicies = set([
            int(getBaseFileName(outFilePath).split('_')[-1])
            for outFilePath in getFilePaths(getExistingSlurmOutputsPath(), '.out')
        ])
        missingOutputIndicies = sorted(set(range(getCountingScriptCount())) - outFileIndicies)

        print('Missing OUT files: {}'.format(len(missingOutputIndicies)))

        return missingOutputIndicies

    def writeNewSubmittingScriptFile():
        missingOutputIndicies = [str(missingOutputIndex) for missingOutputIndex in getMissingOutputIndicies()]

        if missingOutputIndicies:
            missingOutputIndiciesStr = ','.join(missingOutputIndicies)

            slurmArrayCommand = ' '.join([
                'sbatch',
                '--output={}/%A_%a.out'.format(getExistingSlurmOutputsPath()),
                '--array={}'.format(missingOutputsIndiciesStr),
                '--export=countingScriptsPath={},ALL'.format(getExistingCountingScriptsPath()),
                SLURM_ARRAY_SBATCH_PATH,
            ])

            submittingScriptName = 'sNew.sh'
            submittingScriptPath = os.path.join(getExistingSubmittingScriptsPath(), submittingScriptName)
            with open(submittingScriptPath, 'w') as submittingScript:
                submittingScript.write('\n{}\n\n'.format(slurmArrayCommand))
                os.chmod(submittingScriptPath, stat.S_IRUSR ^ stat.S_IWUSR ^ stat.S_IXUSR) # owner reads/writes/executes

            print('Wrote new submitting script:\n\t{}'.format(submittingScriptPath))

        print()

    writeNewSubmittingScriptFile()

def noteMissingSlurmOutputsInDirs():
    outDirs = [
    ]

    for outDir in outDirs:
        noteMissingSlurmOutputs(outDir)

def noteMissingRedirectionOutputs(outDir, commandsPerJob):
    outputs = {} # jobIndex |-> {commandIndex,...}
    outFilePaths = getFilePaths(outDir, '.out')
    for outFilePath in outFilePaths:
        baseName = getBaseFileName(outFilePath)
        (jobIndex, outputIndex) = baseName.split('-')
        jobIndex = int(jobIndex)
        outputIndex = int(outputIndex)
        if jobIndex in outputs:
            outputs[jobIndex].add(outputIndex)
        else:
            outputs[jobIndex] = {outputIndex}

    print('Total: {} jobs, {} outputs'.format(len(outputs), len(outFilePaths)))

    print('Potentially missing outputs:')
    good = True
    for jobIndex in sorted(outputs):
        outputCount = len(outputs[jobIndex])
        if outputCount < commandsPerJob:
            good = False
            print('\tJob:\t{}\tOutputs:\t{}'.format(jobIndex, outputCount))
    if good:
        print('\tNothing')

def mainNotes():
    noteBenchmarkVarsClauses()
    noteMavcs()

    #NOTE analysis/texts/notes.txt:
    # noteOverflows()
    # noteReasonerTime()
    # noteModelCountDiffs(BenchmarkFam.ALTOGETHER, WeightFormat.WEIGHTED)

    # noteCachetDummyModelCounts() # analysis/texts/dummies.txt

    # noteBayesModelCountDistribution()

    # noteMissingSlurmOutputsInDirs()

    print()

################################################################################
'''Handles files.'''

def mainFileRenaming():
    destDir = OUT_DIR_PATH_EXP_1_ADDMC_BAYES_PSEUDOWEIGHTED
    srcDir = os.path.join(destDir, 'redirection')
    for outFilePath in getFilePaths(srcDir):
        namedClusteringHeuristic = getNamedHeuristicTuple(getAddmcOutFileReport(outFilePath, Experiment.ALL_CONFIGS).heuristicTuple)[0]
        os.makedirs(namedClusteringHeuristic, exist_ok=True)
        newOutFilePath = os.path.join(destDir, namedClusteringHeuristic, getFileName(outFilePath))
        os.rename(outFilePath, newOutFilePath)

def mainFileRewriting(): # anonymizes files
    srcDir = DATA_PATH
    destDir = srcDir + '--REWRITTEN'

    replacements = { # 'srcStr': 'destStr'
    }

    srcFilePaths = getFilePaths(srcDir, fileExtension='')
    N = len(srcFilePaths)
    for i in range(N):
        srcFilePath = srcFilePaths[i]

        destFilePath = srcFilePath.replace(srcDir, destDir)
        os.makedirs(os.path.dirname(destFilePath) , exist_ok=True)

        with open(srcFilePath) as srcFile:
            with open(destFilePath, 'w+') as destFile:
                for line in srcFile:
                    for (srcStr, destStr) in replacements.items():
                        line = re.sub(srcStr, destStr, line)
                    destFile.write(line)

        if i % 1e4 == 0:
            print('{}% done'.format(FLOAT_FORMAT.format(100 * i / N)))

################################################################################

def main():
    pass

    # print('\nFlag SHOWING_WARNINGS = {}'.format(SHOWING_WARNINGS))

    # mainSlurmArray() # must check SlurmArray.sbatch

    # mainDataCompression()

    # mainAnalysis()
    # mainPlotting()
    # mainNotes()

    # mainFileRenaming()
    # mainFileRewriting()

if __name__ == '__main__':
    showingTimes = True

    startTime = getTime()
    if showingTimes:
        print('\nStart time:\t{}'.format(startTime), file=sys.stderr)

    print()
    printSeparator()

    main()

    printSeparator()

    endTime = getTime()
    if showingTimes:
        print('\nEnd time:\t{}\n'.format(endTime), file=sys.stderr)
        print('Duration:\t{}\n'.format(endTime - startTime), file=sys.stderr)
