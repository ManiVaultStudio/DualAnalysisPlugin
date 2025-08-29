#include "DualAnalysisPlugin.h"

#include "tSNE/TsneSettingsAction.h"

#include "HSNE/HsneParameters.h"
#include "HSNE/HsneRecomputeWarningDialog.h"
#include "HSNE/HsneScaleAction.h"

#include <PointData/DimensionsPickerAction.h>
#include <PointData/InfoAction.h>
#include <PointData/PointData.h>

#include <event/Event.h>
#include <util/Icon.h>

#include <actions/PluginTriggerAction.h>

#include "hdi/data/io.h"
#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"

#include "hdi/dimensionality_reduction/hierarchical_sne.h"

#include <chrono>

#include <fstream>

Q_PLUGIN_METADATA(IID "nl.tudelft.DualAnalysisPlugin")

using namespace mv;
using namespace mv::util;

DualAnalysisPlugin::DualAnalysisPlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory),
    _settingsAction(this, "Settings Action"), // For B

    _tsneAnalysisB(),
    _tsneSettingsActionB(nullptr),
    _dataPreparationTaskB(this, "Prepare data B"),
    _probDistMatrixB(),

    _tsneAnalysisA(),
    _tsneSettingsActionA(nullptr),
    _dataPreparationTaskA(this, "Prepare data A"),
    _probDistMatrixA(),

    _1DtsneAnalysisA(),
    _1DtsneSettingsActionA(nullptr),
    _1DdataPreparationTaskA(this, "Prepare data A"),
    _1DprobDistMatrixA(),

    _1DtsneAnalysisB(),
    _1DtsneSettingsActionB(nullptr),
    _1DdataPreparationTaskB(this, "Prepare data A"),
    _1DprobDistMatrixB(),

    _recomputedEmbedding2DDatasetsA(),
    _recomputedEmbedding1DDatasetsA(),
    _refinedDatasetsA(),
    _embedding1DRefinedDatasets(),

    _hierarchy(std::make_unique<HsneHierarchy>()),
    _hierarchyThread(),
    _tsneAnalysisHSNEB(),
    _hsneSettingsAction(nullptr),
    _selectionHelperData(nullptr)
{
    setObjectName("Dual Analysis");

    _dataPreparationTaskB.setDescription("All operations prior to TSNE computations");

    qDebug() << "DualAnalysisPlugin created, this pointer:" << this;
}

DualAnalysisPlugin::~DualAnalysisPlugin()
{

    _tsneSettingsActionA->getComputationAction().getStopComputationAction().trigger();

    if (_settingsAction.getEmbeddingAlgorithmAction().getCurrentText() == "tSNE")
    {
        _tsneSettingsActionB->getComputationAction().getStopComputationAction().trigger();
        
    }
    else if (_settingsAction.getEmbeddingAlgorithmAction().getCurrentText() == "HSNE")
    {
        _hierarchyThread.quit();           // Signal the thread to quit gracefully
        if (!_hierarchyThread.wait(500))   // Wait for the thread to actually finish
            _hierarchyThread.terminate();  // Terminate thread after 0.5 seconds
	}
   
    qDebug() << "DualAnalysisPlugin destroyed, this pointer:" << this;
}

void DualAnalysisPlugin::init()
{
    _datasetB = getInputDataset<Points>();// assume the input dataset is a cell by gene matrix
    qDebug() << "input dataset B: " << _datasetB->getGuiName();
	if (!outputDataInit())
	{
        qDebug() << "output data initialization failed";
		transposeData();

        initializeEmbeddingA();

        initializeEmbeddingB();
	}
}

/******************************************************************************
 * data transpose
 ******************************************************************************/

void DualAnalysisPlugin::transposeData()
{
    const auto inputPoints = getInputDataset<Points>();
    qDebug() << "input dataset: " << inputPoints->getGuiName();

    // Retrieve the number of points and dimensions
    const int64_t numPoints = inputPoints->getNumPoints();
    const int64_t numDimensions = inputPoints->getNumDimensions();
    qDebug() << "numPoints: " << numPoints << " numDimensions: " << numDimensions;

    if (numPoints*numDimensions > std::numeric_limits<int64_t>::max())
    {
        throw std::overflow_error("ERROR: numPoints * numDimensions overflows int64_t");
    }

    // Create a vector to store the transposed data
    QVector<float> transposedData(numPoints * numDimensions);//float
    //QVector<biovault::bfloat16_t> transposedData(numPoints * numDimensions);//bfloat

    qDebug() << "transposedData vector initialized";

    // Transposing the data
#pragma omp parallel for
    for (int64_t i = 0; i < numPoints; ++i)
    {
        for (int64_t j = 0; j < numDimensions; ++j)
        {
            const size_t idxInput = static_cast<size_t>(i) * static_cast<size_t>(numDimensions)+ static_cast<size_t>(j);

            transposedData[j * numPoints + i] = inputPoints->getValueAt(idxInput);//float         
            //transposedData[j * numPoints + i] = static_cast<biovault::bfloat16_t>(inputPoints->getValueAt(idxInput));//bfloat
        }

        // Progress reporting
        if (i % 10000 == 0)
        {
			qDebug() << "Transposing data: " << i << " / " << numPoints;
        }

    }
    qDebug() << "transposed vector generated";

    _datasetA = mv::data().createDataset<Points>("Points", "Transposed Data");

    setOutputDataset(_datasetA);

    // Assign the transposed data to the output dataset
    _datasetA->setData(transposedData.data(), numDimensions, numPoints);  // Note the swapped dimensions

    // Inform the core (and thus others) that the data changed
    events().notifyDatasetDataChanged(_datasetA);

    qDebug() << "Transposed data created " << _datasetA->getGuiName();

}

/******************************************************************************
 * output data initialization
 ******************************************************************************/
void DualAnalysisPlugin::initializeEmbeddingA()
{
    // for now, only tsne
    
    if (!_loadingFromProject)
    {
        // initialize 2D embedding A dataset
        auto derivedDataA = mv::data().createDerivedDataset("2D Embedding A", _datasetA, _datasetA);
        _embedding2DDatasetA = Dataset<Points>(derivedDataA.get<Points>());
        setOutputDataset(_embedding2DDatasetA);
        qDebug() << "output dataset created" << _embedding2DDatasetA->getGuiName();

        std::vector<float> initialDataA;
        initialDataA.resize(2 * _datasetA->getNumPoints());

        _embedding2DDatasetA->setData(initialDataA.data(), _datasetA->getNumPoints(), 2);

        events().notifyDatasetDataChanged(_embedding2DDatasetA);
    }

    _embedding2DDatasetA->_infoAction->collapse();

    //_tsneSettingsActionA = new TsneSettingsAction(this, _datasetA->getNumPoints()); // experiment 02.12
    _tsneSettingsActionA = new TsneSettingsAction(this, _datasetB->getNumDimensions());

    setup2DTsneForDataset(_datasetA, _embedding2DDatasetA, _tsneAnalysisA, _tsneSettingsActionA, _probDistMatrixA, _dataPreparationTaskA);

    if (!_loadingFromProject)
    {
        // initialize 1D embedding A dataset
        auto derivedData1DA = mv::data().createDerivedDataset("1D Embedding A", _embedding2DDatasetA, _embedding2DDatasetA);
        _embedding1DDatasetA = Dataset<Points>(derivedData1DA.get<Points>());
        setOutputDataset(_embedding1DDatasetA);
        qDebug() << "output dataset created" << _embedding1DDatasetA->getGuiName();

        std::vector<float> initialData1DA;
        //initialData1DA.resize(_datasetA->getNumPoints());// experiment 02.12
        initialData1DA.resize(_embedding2DDatasetA->getNumPoints());

        _embedding1DDatasetA->setData(initialData1DA.data(), _datasetA->getNumPoints(), 1);
        events().notifyDatasetDataChanged(_embedding1DDatasetA);
    }

    _1DtsneSettingsActionA = new TsneSettingsAction(this, _embedding2DDatasetA->getNumPoints());

    connect(&_tsneAnalysisA, &TsneAnalysis::finished, this, [this]() {
        _embedding2DA = convertDatasetToEmbedding(_embedding2DDatasetA, 2);

        if (_triggerAlignment == 1) {
            qDebug() << "Hard coded here to make it not compute 1D again"; // FIXME
            return;
        }
        compute1DTsne(_embedding2DDatasetA, _embedding1DDatasetA, _1DtsneAnalysisA, _1DtsneSettingsActionA, _1DdataPreparationTaskA, _embedding2DA);
        });
}

void DualAnalysisPlugin::initializeEmbeddingB()
{
	if (!_loadingFromProject)
	{
		// initialize 2D embedding B dataset - can choose from tsne or hsne
		auto derivedDataB = mv::data().createDerivedDataset("2D Embedding B", _datasetB, _datasetB);
		_embedding2DDatasetB = Dataset<Points>(derivedDataB.get<Points>());
		setOutputDataset(_embedding2DDatasetB);
		qDebug() << "output dataset created" << _embedding2DDatasetB->getGuiName();

		_embedding2DDatasetB->setData(nullptr, _datasetB->getNumPoints(), 2);
		events().notifyDatasetDataChanged(_embedding2DDatasetB);

		_embedding2DDatasetB->getDataHierarchyItem().select(true);
		_embedding2DDatasetB->_infoAction->collapse();

		_embedding2DDatasetB->addAction(_settingsAction);
		_settingsAction.expand();

		connect(&_settingsAction.getEmbeddingAlgorithmAction(), &OptionAction::currentTextChanged, this, [this](const QString& value) {
			if (value == "tSNE")
			{
				qDebug() << "tSNE selected";

				// enable tSNE settings
				_tsneSettingsActionB = new TsneSettingsAction(this, _datasetB->getNumPoints());
				setup2DTsneForDataset(_datasetB, _embedding2DDatasetB, _tsneAnalysisB, _tsneSettingsActionB, _probDistMatrixB, _dataPreparationTaskB);

				if (_hsneSettingsAction != nullptr)
				{
					qDebug() << "Warning! Hsne Setting is already initialized";
					if (_hsneSettingsAction->isVisible())
						_hsneSettingsAction->setVisible(false);
					// FIXME: any further action for safety?    
				}
				_settingsAction.getEmbeddingAlgorithmAction().setEnabled(false);// TODO: assume choice is only made once           
			}
			else if (value == "HSNE")
			{
				qDebug() << "HSNE selected"; // TODO

				// enable HSNE settings
				_hsneSettingsAction = new HsneSettingsAction(this);
				setupHSNEForDataset(_datasetB, _embedding2DDatasetB, _tsneAnalysisHSNEB);

				if (_tsneSettingsActionB != nullptr)
				{
					qDebug() << "Warning! Tsne Setting is already initialized";
					if (_tsneSettingsActionB->isVisible())
						_tsneSettingsActionB->setVisible(false);
					// FIXME: any further action for safety?
				}

				_settingsAction.getEmbeddingAlgorithmAction().setEnabled(false);
			}
			});

		// initialize 1D embedding B dataset
		auto derivedData1DB = mv::data().createDerivedDataset("1D Embedding B", _embedding2DDatasetB, _embedding2DDatasetB);
		_embedding1DDatasetB = Dataset<Points>(derivedData1DB.get<Points>());
		setOutputDataset(_embedding1DDatasetB);
		qDebug() << "output dataset created" << _embedding1DDatasetB->getGuiName();

		std::vector<float> initialData1DB;
		initialData1DB.resize(_embedding2DDatasetB->getNumPoints());

		_embedding1DDatasetB->setData(initialData1DB.data(), _embedding2DDatasetB->getNumPoints(), 1);
		events().notifyDatasetDataChanged(_embedding1DDatasetB);
	}

	_1DtsneSettingsActionB = new TsneSettingsAction(this, _embedding2DDatasetB->getNumPoints());

	// if 2D embedding B is tsne
	connect(&_tsneAnalysisB, &TsneAnalysis::finished, this, [this]() {
		_embedding2DB = convertDatasetToEmbedding(_embedding2DDatasetB, 2);

		compute1DTsne(_embedding2DDatasetB, _embedding1DDatasetB, _1DtsneAnalysisB, _1DtsneSettingsActionB, _1DdataPreparationTaskB, _embedding2DB);
		});

	// if 2D embedding B is hsne
	connect(&_tsneAnalysisHSNEB, &TsneAnalysis::finished, this, [this]() {
		_embedding2DB = convertDatasetToEmbedding(_embedding2DDatasetB, 2);

        // temp FIX: to change the size of the 1D embedding to the size of the 2D embedding (num landmarks)
        std::vector<float> initialData1DB;
        initialData1DB.resize(_embedding2DDatasetB->getNumPoints());
        _embedding1DDatasetB->setData(initialData1DB.data(), _embedding2DDatasetB->getNumPoints(), 1);
        events().notifyDatasetDataChanged(_embedding1DDatasetB);

		compute1DTsne(_embedding2DDatasetB, _embedding1DDatasetB, _1DtsneAnalysisB, _1DtsneSettingsActionB, _1DdataPreparationTaskB, _embedding2DB);
		});

	connect(&_settingsAction.getAlignmentAction(), &TriggerAction::triggered, this, [this]() {

		onAlignmentTriggered();
		});
}

/******************************************************************************
 * 2D tSNE computation
 ******************************************************************************/

void DualAnalysisPlugin::setup2DTsneForDataset(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, 
    TsneSettingsAction*& tsneSettingsAction, ProbDistMatrix& probDistMatrix, mv::Task& dataPreparationTask)
{

    dataPreparationTask.setParentTask(&embeddingDataset->getTask());

    embeddingDataset->addAction(tsneSettingsAction->getGeneralTsneSettingsAction());
    //embeddingDataset->addAction(tsneSettingsAction->getInitalEmbeddingSettingsAction());
    //embeddingDataset->addAction(tsneSettingsAction->getGradientDescentSettingsAction());
    //embeddingDataset->addAction(tsneSettingsAction->getKnnSettingsAction());
    //auto dimensionsGroupAction = new GroupAction(this, "Dimensions", true);
    //dimensionsGroupAction->addAction(&inputDataset->getFullDataset<Points>()->getDimensionsPickerAction());
    //dimensionsGroupAction->setText(QString("Input dimensions (%1)").arg(inputDataset->getFullDataset<Points>()->text()));
    //dimensionsGroupAction->setShowLabels(false);
    //embeddingDataset->addAction(*dimensionsGroupAction);

    // update settings that depend on number of data points
    tsneSettingsAction->getGradientDescentSettingsAction().getExaggerationFactorAction().setValue(4.f + inputDataset->getNumPoints() / 60000.0f);

    auto& computationAction = tsneSettingsAction->getComputationAction();

    const auto updateComputationAction = [this, &tsneAnalysis, &computationAction]() {
        const auto isRunning = computationAction.getRunningAction().isChecked();

        computationAction.getStartComputationAction().setEnabled(!isRunning);
        computationAction.getContinueComputationAction().setEnabled(!isRunning && tsneAnalysis.canContinue());
        computationAction.getStopComputationAction().setEnabled(isRunning);
        };

    auto changeSettingsReadOnly = [this, &tsneSettingsAction](bool readonly) -> void {
        tsneSettingsAction->getGeneralTsneSettingsAction().setReadOnly(readonly);
        //tsneSettingsAction->getInitalEmbeddingSettingsAction().setReadOnly(readonly);
        //tsneSettingsAction->getGradientDescentSettingsAction().setReadOnly(readonly);
        //tsneSettingsAction->getKnnSettingsAction().setReadOnly(readonly);
        };

    connect(&tsneAnalysis, &TsneAnalysis::finished, this, [this, &computationAction, changeSettingsReadOnly]() {
        computationAction.getRunningAction().setChecked(false);

        changeSettingsReadOnly(false);
        });

    connect(&tsneAnalysis, &TsneAnalysis::aborted, this, [this, &computationAction, updateComputationAction, changeSettingsReadOnly]() {
        updateComputationAction();

        computationAction.getRunningAction().setChecked(false);

        changeSettingsReadOnly(false);
        });

    connect(&computationAction.getStartComputationAction(), &TriggerAction::triggered, this, [this, &inputDataset, &embeddingDataset, &tsneAnalysis, &tsneSettingsAction, &probDistMatrix, &dataPreparationTask , &computationAction, changeSettingsReadOnly]() {
        qDebug() << "startComputation triggered";

        changeSettingsReadOnly(true);

        if (tsneSettingsAction->getGeneralTsneSettingsAction().getReinitAction().isChecked())
            reinitializeComputation(embeddingDataset, tsneAnalysis, tsneSettingsAction, probDistMatrix);
        else
            startComputation(inputDataset, embeddingDataset, tsneAnalysis, tsneSettingsAction, dataPreparationTask);

        tsneSettingsAction->getGeneralTsneSettingsAction().getReinitAction().setCheckable(true);   // only enable re-init after first computation
        });

    connect(&computationAction.getContinueComputationAction(), &TriggerAction::triggered, this, [this, &embeddingDataset, &tsneAnalysis, &tsneSettingsAction, &probDistMatrix, &dataPreparationTask, changeSettingsReadOnly]() {
        changeSettingsReadOnly(true);

        continueComputation(embeddingDataset, tsneAnalysis, tsneSettingsAction, probDistMatrix, dataPreparationTask);
        });

    connect(&computationAction.getStopComputationAction(), &TriggerAction::triggered, this, [this, &tsneAnalysis, &tsneSettingsAction]() {
        qApp->processEvents();

        stopComputation(tsneAnalysis);
        });

    connect(&tsneAnalysis, &TsneAnalysis::embeddingUpdate, this, [this, &tsneAnalysis, &tsneSettingsAction, &embeddingDataset](const TsneData tsneData) {

        // Update the output points dataset with new data from the TSNE analysis
        embeddingDataset->setData(tsneData.getData().data(), tsneData.getNumPoints(), tsneSettingsAction->getGeneralTsneSettingsAction().getNumDimensionOutputAction().getCurrentText().toInt());

        tsneSettingsAction->getGeneralTsneSettingsAction().getNumberOfComputatedIterationsAction().setValue(tsneAnalysis.getNumIterations() - 1);

        // Notify others that the embedding data changed
        events().notifyDatasetDataChanged(embeddingDataset);
        });

    connect(&computationAction.getRunningAction(), &ToggleAction::toggled, this, [this, &computationAction, &inputDataset, updateComputationAction](bool toggled) {
        inputDataset->getDimensionsPickerAction().setEnabled(!toggled);
        updateComputationAction();
        });

    updateComputationAction();

    auto& datasetTask = embeddingDataset->getTask();

    datasetTask.setName("Compute 2D TSNE");
    datasetTask.setConfigurationFlag(Task::ConfigurationFlag::OverrideAggregateStatus);

    tsneAnalysis.setTask(&datasetTask);

   /* _eventListener.addSupportedEventType(static_cast<std::uint32_t>(EventType::DatasetDataChanged));
    _eventListener.registerDataEvent([this, &tsneSettingsAction](DatasetEvent* dataEvent) {
        const auto& dataset = dataEvent->getDataset();

        if (dataset->getDataType() != PointType)
            return;

        tsneSettingsAction->getInitalEmbeddingSettingsAction().updateDatasetPicker();

    });*/

    connect(&embeddingDataset, &Dataset<Points>::dataChanged, this, [this, &tsneSettingsAction]()
        {
            tsneSettingsAction->getInitalEmbeddingSettingsAction().updateDatasetPicker(); // TODO: is this affecting the alignment computation? i.e. we do not want to reinitialize the embedding when alignment is triggered
             
		});

    qDebug() << "tSNE setup done for " << inputDataset->getGuiName();
}

void DualAnalysisPlugin::startComputation(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction, mv::Task& dataPreparationTask)
{
    embeddingDataset->getTask().setRunning();

    dataPreparationTask.setEnabled(true);
    dataPreparationTask.setRunning();

    // Create list of data from the enabled dimensions
    std::vector<float> data;
    std::vector<unsigned int> indices;

    // Extract the enabled dimensions from the data
    std::vector<bool> enabledDimensions = inputDataset->getDimensionsPickerAction().getEnabledDimensions();

    const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    tsneSettingsAction->getGeneralTsneSettingsAction().getNumberOfComputatedIterationsAction().reset();

    const auto numPoints = inputDataset->isFull() ? inputDataset->getNumPoints() : inputDataset->indices.size();
    data.resize(numPoints * numEnabledDimensions);

    for (int i = 0; i < inputDataset->getNumDimensions(); i++)
        if (enabledDimensions[i])
            indices.push_back(i);

    inputDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, indices);

    tsneSettingsAction->getComputationAction().getRunningAction().setChecked(true);

    // Init embedding: random or set from other dataset, e.g. PCA
    auto initEmbedding = tsneSettingsAction->getInitalEmbeddingSettingsAction().getInitEmbedding(numPoints); // random initialization

    dataPreparationTask.setFinished();

    tsneAnalysis.startComputation(tsneSettingsAction->getTsneParameters(), tsneSettingsAction->getKnnParameters(), std::move(data), numEnabledDimensions, &initEmbedding);
}

void DualAnalysisPlugin::startComputation(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction, mv::Task& dataPreparationTask,
    hdi::data::Embedding<float>& masterEmbedding)
{
    embeddingDataset->getTask().setRunning();

    dataPreparationTask.setEnabled(true);
    dataPreparationTask.setRunning();

    // Create list of data from the enabled dimensions
    std::vector<float> data;
    std::vector<unsigned int> indices;

    // Extract the enabled dimensions from the data
    std::vector<bool> enabledDimensions = inputDataset->getDimensionsPickerAction().getEnabledDimensions();

    const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    tsneSettingsAction->getGeneralTsneSettingsAction().getNumberOfComputatedIterationsAction().reset();

    const auto numPoints = inputDataset->isFull() ? inputDataset->getNumPoints() : inputDataset->indices.size();
    data.resize(numPoints * numEnabledDimensions);

    for (int i = 0; i < inputDataset->getNumDimensions(); i++)
        if (enabledDimensions[i])
            indices.push_back(i);

    inputDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, indices);

    tsneSettingsAction->getComputationAction().getRunningAction().setChecked(true);

    // Init embedding: random or set from other dataset, e.g. PCA
    //auto initEmbedding = tsneSettingsAction->getInitalEmbeddingSettingsAction().getInitEmbedding(numPoints);

    qDebug() << "Set y axis of 2D embedding as the init embedding of 1D embedding";
    std::vector<float> initEmbedding(embeddingDataset->getNumPoints() * embeddingDataset->getNumDimensions());
    inputDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding, { 1 });
    qDebug() << "initEmbedding size: " << initEmbedding.size() << " " << initEmbedding[0] << " " << initEmbedding[1] << " " << initEmbedding[2] << " " << initEmbedding[3];

    dataPreparationTask.setFinished();

    tsneAnalysis.startComputation(tsneSettingsAction->getTsneParameters(), tsneSettingsAction->getKnnParameters(), std::move(data), numEnabledDimensions, &initEmbedding, &masterEmbedding.getContainer());
}

void DualAnalysisPlugin::reinitializeComputation(mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction, ProbDistMatrix& probDistMatrix)
{
    if (tsneAnalysis.canContinue())
        probDistMatrix = std::move(*tsneAnalysis.getProbabilityDistribution().value());
    
    if(probDistMatrix.size() == 0)
    {
        qDebug() << "DualAnalysisPlugin::reinitializeComputation: cannot reinitialize embedding - start computation first";
        return;
    }

    auto& initSettings = tsneSettingsAction->getInitalEmbeddingSettingsAction();

    if (initSettings.getRandomInitAction().isChecked() && initSettings.getNewRandomSeedAction().isChecked())
        initSettings.updateSeed();

    const auto numPoints = embeddingDataset->getNumPoints();

    auto initEmbedding = initSettings.getInitEmbedding(numPoints);

    tsneAnalysis.startComputation(tsneSettingsAction->getTsneParameters(), std::move(probDistMatrix), numPoints, &initEmbedding);
}

void DualAnalysisPlugin::continueComputation(mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction, ProbDistMatrix& probDistMatrix, mv::Task& dataPreparationTask)
{
    embeddingDataset->getTask().setRunning();

    dataPreparationTask.setEnabled(false);

    tsneSettingsAction->getComputationAction().getRunningAction().setChecked(true);

    if (tsneAnalysis.canContinue())
        tsneAnalysis.continueComputation(tsneSettingsAction->getTsneParameters().getNumIterations());
    else if (probDistMatrix.size() > 0)
    {
        auto currentEmbedding = embeddingDataset;

        std::vector<float> currentEmbeddingPositions;
        currentEmbeddingPositions.resize(2ull * currentEmbedding->getNumPoints()); // FIXME: 2ull????
        currentEmbedding->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(currentEmbeddingPositions, { 0, 1 });

        tsneAnalysis.startComputation(tsneSettingsAction->getTsneParameters(), std::move(probDistMatrix), currentEmbedding->getNumPoints(), &currentEmbeddingPositions, tsneSettingsAction->getGeneralTsneSettingsAction().getNumberOfComputatedIterationsAction().getValue());
    }
    else
    {
        qWarning() << "DualAnalysisPlugin::continueComputation: cannot continue.";
        tsneSettingsAction->getComputationAction().getRunningAction().setChecked(false);
        dataPreparationTask.setEnabled(false);
        embeddingDataset->getTask().setFinished();
    }
}

void DualAnalysisPlugin::stopComputation(TsneAnalysis& tsneAnalysis)
{
    tsneAnalysis.stopComputation();
}

/******************************************************************************
 * 1D tSNE computation
 ******************************************************************************/

void DualAnalysisPlugin::compute1DTsne(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsne1DSettingsAction, mv::Task& dataPreparationTask)
{
    tsne1DSettingsAction->getGeneralTsneSettingsAction().getNumDimensionOutputAction().setCurrentIndex(0); // set 1D output

    tsne1DSettingsAction->getGeneralTsneSettingsAction().getNumIterationsAction().setValue(500); // TODO: set iterations

    // TODO: set theta 

    dataPreparationTask.setParentTask(&embeddingDataset->getTask());

    auto& datasetTask = embeddingDataset->getTask();

    datasetTask.setName("Compute 1D TSNE");
    datasetTask.setConfigurationFlag(Task::ConfigurationFlag::OverrideAggregateStatus);

    tsneAnalysis.setTask(&datasetTask);

    _eventListener.addSupportedEventType(static_cast<std::uint32_t>(EventType::DatasetDataChanged));
    _eventListener.registerDataEvent([this, &tsne1DSettingsAction](DatasetEvent* dataEvent) {
        const auto& dataset = dataEvent->getDataset();

        if (dataset->getDataType() != PointType)
            return;

        tsne1DSettingsAction->getInitalEmbeddingSettingsAction().updateDatasetPicker();

        });

    connect(&tsneAnalysis, &TsneAnalysis::embeddingUpdate, this, [this, &tsneAnalysis, &tsne1DSettingsAction, &embeddingDataset](const TsneData tsneData) {

        // Update the output points dataset with new data from the TSNE analysis
        embeddingDataset->setData(tsneData.getData().data(), tsneData.getNumPoints(), tsne1DSettingsAction->getGeneralTsneSettingsAction().getNumDimensionOutputAction().getCurrentText().toInt());

        tsne1DSettingsAction->getGeneralTsneSettingsAction().getNumberOfComputatedIterationsAction().setValue(tsneAnalysis.getNumIterations() - 1);

        // Notify others that the embedding data changed
        events().notifyDatasetDataChanged(embeddingDataset);
        });

    startComputation(inputDataset, embeddingDataset, tsneAnalysis, tsne1DSettingsAction, dataPreparationTask);

}


void DualAnalysisPlugin::compute1DTsne(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsne1DSettingsAction, mv::Task& dataPreparationTask,
    hdi::data::Embedding<float>& masterEmbedding)
{
    tsne1DSettingsAction->getGeneralTsneSettingsAction().getNumDimensionOutputAction().setCurrentIndex(0); // set 1D output

    tsne1DSettingsAction->getGeneralTsneSettingsAction().getNumIterationsAction().setValue(500); // TODO: set iterations

    tsne1DSettingsAction->getGeneralTsneSettingsAction().getPerplexityAction().setValue(30);// TODO: set perplexity

    // TODO: set theta 

    dataPreparationTask.setParentTask(&embeddingDataset->getTask());

    auto& datasetTask = embeddingDataset->getTask();

    datasetTask.setName("Compute 1D TSNE using master");
    datasetTask.setConfigurationFlag(Task::ConfigurationFlag::OverrideAggregateStatus);

    tsneAnalysis.setTask(&datasetTask);

    _eventListener.addSupportedEventType(static_cast<std::uint32_t>(EventType::DatasetDataChanged));
    _eventListener.registerDataEvent([this, &tsne1DSettingsAction](DatasetEvent* dataEvent) {
        const auto& dataset = dataEvent->getDataset();

        if (dataset->getDataType() != PointType)
            return;

        tsne1DSettingsAction->getInitalEmbeddingSettingsAction().updateDatasetPicker();

        });

    connect(&tsneAnalysis, &TsneAnalysis::embeddingUpdate, this, [this, &tsneAnalysis, &tsne1DSettingsAction, &embeddingDataset](const TsneData tsneData) {

        // Update the output points dataset with new data from the TSNE analysis
        embeddingDataset->setData(tsneData.getData().data(), tsneData.getNumPoints(), tsne1DSettingsAction->getGeneralTsneSettingsAction().getNumDimensionOutputAction().getCurrentText().toInt());

        tsne1DSettingsAction->getGeneralTsneSettingsAction().getNumberOfComputatedIterationsAction().setValue(tsneAnalysis.getNumIterations() - 1);

        // Notify others that the embedding data changed
        events().notifyDatasetDataChanged(embeddingDataset);
        });

    startComputation(inputDataset, embeddingDataset, tsneAnalysis, tsne1DSettingsAction, dataPreparationTask, masterEmbedding);

}


hdi::data::Embedding<float> DualAnalysisPlugin::convertDatasetToEmbedding(const mv::Dataset<Points>& dataset, int dimensionality) {
    const auto numPoints = dataset->getNumPoints();
    std::vector<float> data(numPoints * dimensionality);

    if (dimensionality == 2)
       dataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, { 0, 1 }); 
    if (dimensionality == 1)
        dataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, { 0 });

    hdi::data::Embedding<float> embedding(dimensionality, numPoints);
    embedding.getContainer() = std::move(data);

    return embedding;
}


/******************************************************************************
 * HSNE computation
 ******************************************************************************/
void DualAnalysisPlugin::setupHSNEForDataset(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis)
{
    int numHierarchyScales = std::max(1L, std::lround(log10(inputDataset->getNumPoints())) - 2);
    _hsneSettingsAction->getGeneralHsneSettingsAction().getNumScalesAction().setValue(numHierarchyScales);

    embeddingDataset->addAction(_hsneSettingsAction->getGeneralHsneSettingsAction());
    embeddingDataset->addAction(_hsneSettingsAction->getHierarchyConstructionSettingsAction());
    embeddingDataset->addAction(_hsneSettingsAction->getTopLevelScaleAction());
    embeddingDataset->addAction(_hsneSettingsAction->getGradientDescentSettingsAction());
    embeddingDataset->addAction(_hsneSettingsAction->getKnnSettingsAction());

    auto dimensionsGroupAction = new GroupAction(this, "Dimensions", true);
    dimensionsGroupAction->addAction(&inputDataset->getFullDataset<Points>()->getDimensionsPickerAction());
    dimensionsGroupAction->setText(QString("Input dimensions (%1)").arg(inputDataset->getFullDataset<Points>()->text()));
    dimensionsGroupAction->setShowLabels(false);
    embeddingDataset->addAction(*dimensionsGroupAction);

    inputDataset->setProperty("selectionHelperCount", 0);

    // update settings that depend on number of data points
    _hsneSettingsAction->getGradientDescentSettingsAction().getExaggerationFactorAction().setValue(4.f + inputDataset->getNumPoints() / 60000.0f);

    auto& computationAction = _hsneSettingsAction->getTopLevelScaleAction().getComputationAction();

    const auto updateComputationAction = [this, &tsneAnalysis, &computationAction]() {
        const auto isRunning = computationAction.getRunningAction().isChecked();

        computationAction.getStartComputationAction().setEnabled(!isRunning);
        computationAction.getContinueComputationAction().setEnabled(!isRunning && tsneAnalysis.canContinue());
        computationAction.getStopComputationAction().setEnabled(isRunning);
        };

    connect(&tsneAnalysis, &TsneAnalysis::finished, this, [this, &computationAction, updateComputationAction]() {
        computationAction.getRunningAction().setChecked(false);

        _hsneSettingsAction->setReadOnly(false);

        updateComputationAction();
        });

    connect(&tsneAnalysis, &TsneAnalysis::aborted, this, [this, &computationAction, updateComputationAction]() {
        computationAction.getRunningAction().setChecked(false);

        _hsneSettingsAction->setReadOnly(false);

        updateComputationAction();
        });

    connect(&computationAction.getStartComputationAction(), &TriggerAction::triggered, this, [this, &tsneAnalysis, &computationAction]() {
        _hsneSettingsAction->setReadOnly(true);

        int topScaleIndex = _hierarchy->getTopScale();
        Hsne::scale_type& topScale = _hierarchy->getScale(topScaleIndex);
        int numLandmarks = topScale.size();
        TsneParameters tsneParameters = _hsneSettingsAction->getTsneParameters();

        tsneAnalysis.startComputation(tsneParameters, _hierarchy->getTransitionMatrixAtScale(topScaleIndex), numLandmarks);
        });

    connect(&computationAction.getContinueComputationAction(), &TriggerAction::triggered, this, [this, &tsneAnalysis, &embeddingDataset]() {
        _hsneSettingsAction->getGradientDescentSettingsAction().setReadOnly(true);

        continueComputation(embeddingDataset, tsneAnalysis);
        });

    connect(&computationAction.getStopComputationAction(), &TriggerAction::triggered, this, [this, &tsneAnalysis]() {
        qApp->processEvents();

        tsneAnalysis.stopComputation();
        });

    connect(&computationAction.getRunningAction(), &ToggleAction::toggled, this, [this, &inputDataset, updateComputationAction](bool toggled) {
        inputDataset->getDimensionsPickerAction().setEnabled(!toggled);
        updateComputationAction();
        });

    // once HsneHierarchy::initialize is done, it'll emit HsneHierarchy::finished
    connect(this, &DualAnalysisPlugin::startHierarchyWorker, _hierarchy.get(), &HsneHierarchy::initialize);

    connect(_hierarchy.get(), &HsneHierarchy::finished, this, [this, &inputDataset, &embeddingDataset, &tsneAnalysis]() {

        _hsneSettingsAction->getGeneralHsneSettingsAction().setReadOnly(false);
        _hsneSettingsAction->getHierarchyConstructionSettingsAction().setReadOnly(false);
        _hsneSettingsAction->getTopLevelScaleAction().setReadOnly(false);
        _hsneSettingsAction->getGradientDescentSettingsAction().setReadOnly(false);
        _hsneSettingsAction->getKnnSettingsAction().setReadOnly(false);

        _hsneSettingsAction->getGeneralHsneSettingsAction().getStartAction().setText("Recompute");
        _hsneSettingsAction->getGeneralHsneSettingsAction().getStartAction().setToolTip("Recomputing does not change the selection mapping.\n If the data size changed, prefer creating a new HSNE analysis.");

        computeTopLevelEmbedding(inputDataset, embeddingDataset, tsneAnalysis);
        });

    connect(&_hsneSettingsAction->getGeneralHsneSettingsAction().getStartAction(), &TriggerAction::triggered, this, [this, &inputDataset, &embeddingDataset, &tsneAnalysis](bool toggled) {

        // Create a warning dialog if there are already refined scales
        if (_selectionHelperData.isValid() && embeddingDataset->getDataHierarchyItem().getChildren().size() > 0)
        {
            HsneRecomputeWarningDialog dialog;

            if (dialog.exec() == QDialog::Rejected)
                return;
        }

        tsneAnalysis.stopComputation();

        _hsneSettingsAction->getGeneralHsneSettingsAction().setReadOnly(true);
        _hsneSettingsAction->getHierarchyConstructionSettingsAction().setReadOnly(true);
        _hsneSettingsAction->getTopLevelScaleAction().setReadOnly(true);
        _hsneSettingsAction->getGradientDescentSettingsAction().setReadOnly(true);
        _hsneSettingsAction->getTopLevelScaleAction().getComputationAction().getStartComputationAction().setEnabled(false);
        _hsneSettingsAction->getKnnSettingsAction().setReadOnly(true);

        // Initialize the HSNE algorithm with the given parameters and compute the hierarchy
        std::vector<bool> enabledDimensions = inputDataset->getDimensionsPickerAction().getEnabledDimensions();
        _hierarchy->setDataAndParameters(inputDataset, embeddingDataset, _hsneSettingsAction->getHsneParameters(), _hsneSettingsAction->getKnnParameters(), std::move(enabledDimensions));
        _hierarchy->initParentTask();

        _hierarchy->moveToThread(&_hierarchyThread);

        _hierarchyThread.start();
        emit startHierarchyWorker();

        });

    connect(&tsneAnalysis, &TsneAnalysis::started, this, [this, &computationAction, updateComputationAction]() {
        computationAction.getRunningAction().setChecked(true);
        updateComputationAction();
        qApp->processEvents();
        });

    connect(&tsneAnalysis, &TsneAnalysis::embeddingUpdate, this, [this, &embeddingDataset, &tsneAnalysis](const TsneData& tsneData) {
        embeddingDataset->setData(tsneData.getData().data(), tsneData.getNumPoints(), 2);

        _hsneSettingsAction->getTopLevelScaleAction().getNumberOfComputatedIterationsAction().setValue(tsneAnalysis.getNumIterations() - 1);

        // NOTE: Commented out because it causes a stack overflow after a couple of iterations
        //QCoreApplication::processEvents();

        events().notifyDatasetDataChanged(embeddingDataset);
        });

    //_eventListener.addSupportedEventType(static_cast<std::uint32_t>(EventType::DatasetAboutToBeRemoved));
    //_eventListener.registerDataEventByType(PointType, [this, &embeddingDataset](DatasetEvent* dataEvent) {
    connect(&embeddingDataset, &Dataset<Points>::aboutToBeRemoved, this, [this, embeddingDataset]() {
        if (!_selectionHelperData.isValid())
            return;

        qDebug() << "HSNE Plugin: remove (invisible) selection helper dataset " << _selectionHelperData->getId() << " used for deleted " << embeddingDataset->getId();
        mv::data().removeDataset(_selectionHelperData);
     });

    updateComputationAction();

    // Before the hierarchy is initialized, no embedding re-init is possible
    if (!_hierarchy->isInitialized())
        computationAction.getStartComputationAction().setEnabled(false);

    auto& datasetTask = embeddingDataset->getTask();

    tsneAnalysis.setTask(&datasetTask);
}

void DualAnalysisPlugin::computeTopLevelEmbedding(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis)
{
    auto& datasetTask = embeddingDataset->getTask();
    datasetTask.setFinished();
    datasetTask.setName("Compute HSNE top level embedding");
    datasetTask.setRunning();

    // Get the top scale of the HSNE hierarchy
    const int topScaleIndex = _hierarchy->getTopScale();
    Hsne::scale_type& topScale = _hierarchy->getScale(topScaleIndex);
    _hsneSettingsAction->getTopLevelScaleAction().setScale(topScaleIndex);

    _hierarchy->printScaleInfo();

    // Number of landmarks on the top scale
    const uint32_t numLandmarks = topScale.size();


    // Only create new selection helper if a) it does not exist yet and b) we are above the data scale
    if (!_selectionHelperData.isValid() && topScaleIndex > 0)
    {
        // Create a subset of the points corresponding to the top level HSNE landmarks,
        // Then derive the embedding from this subset
        auto selectionDataset = inputDataset->getSelection<Points>();

        // Select the appropriate points to create a subset from
        selectionDataset->indices.resize(numLandmarks);

        if (inputDataset->isFull())
        {
            for (uint32_t i = 0; i < numLandmarks; i++)
                selectionDataset->indices[i] = topScale._landmark_to_original_data_idx[i];
        }
        else
        {
            std::vector<unsigned int> globalIndices;
            inputDataset->getGlobalIndices(globalIndices);
            for (uint32_t i = 0; i < numLandmarks; i++)
                selectionDataset->indices[i] = globalIndices[topScale._landmark_to_original_data_idx[i]];
        }

        // Create the subset and clear the selection
        auto selectionHelperCount = inputDataset->getProperty("selectionHelperCount").toInt();
        inputDataset->setProperty("selectionHelperCount", ++selectionHelperCount);
        _selectionHelperData = inputDataset->createSubsetFromSelection(QString("Hsne selection helper %1").arg(selectionHelperCount), inputDataset, /*visible = */ false);

        selectionDataset->indices.clear();

        embeddingDataset->setSourceDataset(_selectionHelperData);

        // Add linked selection between the upper embedding and the bottom layer
        {
            LandmarkMap& landmarkMap = _hierarchy->getInfluenceHierarchy().getMap()[topScaleIndex];

            mv::SelectionMap mapping;
            auto& selectionMap = mapping.getMap();

            if (inputDataset->isFull())
            {
                std::vector<unsigned int> globalIndices;
                _selectionHelperData->getGlobalIndices(globalIndices);

                for (unsigned int i = 0; i < landmarkMap.size(); i++)
                {
                    selectionMap[globalIndices[i]] = landmarkMap[i];
                }
            }
            else
            {
                std::vector<unsigned int> globalIndices;
                inputDataset->getGlobalIndices(globalIndices);
                for (unsigned int i = 0; i < landmarkMap.size(); i++)
                {
                    std::vector<unsigned int> bottomMap = landmarkMap[i];
                    for (unsigned int j = 0; j < bottomMap.size(); j++)
                    {
                        bottomMap[j] = globalIndices[bottomMap[j]];
                    }
                    auto bottomLevelIdx = _hierarchy->getScale(topScaleIndex)._landmark_to_original_data_idx[i];
                    selectionMap[globalIndices[bottomLevelIdx]] = bottomMap;
                }
            }

            embeddingDataset->addLinkedData(inputDataset, mapping);
        }
    }

    // Set t-SNE parameters
    TsneParameters tsneParameters = _hsneSettingsAction->getTsneParameters();

    // Embed data
    tsneAnalysis.stopComputation();
    tsneAnalysis.startComputation(tsneParameters, _hierarchy->getTransitionMatrixAtScale(topScaleIndex), numLandmarks);
}

void DualAnalysisPlugin::continueComputation(mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis)
{
    embeddingDataset->getTask().setRunning();

    _hsneSettingsAction->getTopLevelScaleAction().getComputationAction().getRunningAction().setChecked(true);

    tsneAnalysis.continueComputation(_hsneSettingsAction->getTsneParameters().getNumIterations());
}


/******************************************************************************
 * Event handling
 ******************************************************************************/
void DualAnalysisPlugin::onRefineFinished(mv::Dataset<Points> refineEmbedding)
{
    qDebug() << "<<<<<<<<DualAnalysisPlugin::onRefineFinished";

    if (!refineEmbedding.isValid()) {
        qWarning() << "DualAnalysisPlugin::onRefineFinished: invalid dataset";
        return;
    }

    qDebug() << "Refine Embedding Dataset:" << refineEmbedding->getGuiName();

    // count the number of recomputed embeddings
    _numRecomputedEmbeddings++;

    // FIXME: generalize this in serialization
    if (_embedding2DDatasetA.isValid())
    {
        if (_embedding2DA.numDataPoints() == 0)// temp FIX to get the embedding data when _embedding2DDatasetA is loaded from project
        {
            _embedding2DA = convertDatasetToEmbedding(_embedding2DDatasetA, 2);
        }
    }
    else
    {
        qDebug() << "onRefineFinished(): embedding2DDatasetA is not valid";
        return;
    }

    // FIXME: is the indexing here correct when hsne scales > 2? 
    // get the transposed matrix of the drilled in source matrix
    auto refinedSourceDatasetB = refineEmbedding->getSourceDataset<Points>();
    const auto refinednumPoints = refinedSourceDatasetB->getNumPoints();
    const auto refinednumDimensions = refinedSourceDatasetB->getNumDimensions();
    QVector<float> transposedData(refinednumPoints * refinednumDimensions);
    for (int i = 0; i < refinednumPoints; ++i)
    {
        for (int j = 0; j < refinednumDimensions; ++j)
        {
            // Correct indexing for the transposed data
            transposedData[j * refinednumPoints + i] = refinedSourceDatasetB->getValueAt(i * refinednumDimensions + j);
        }
    }

    // add a new dataset for the refined transposed data
    _refinedDatasetsA.push_back(mv::data().createDataset<Points>("Points", QString("Refined Transposed Data %1").arg(_numRecomputedEmbeddings)));
    auto& refinedDatasetA = _refinedDatasetsA.back();
    refinedDatasetA->setData(transposedData.data(), refinednumDimensions, refinednumPoints);
    events().notifyDatasetAdded(refinedDatasetA);
    events().notifyDatasetDataChanged(refinedDatasetA);
    qDebug() << "Refined Transposed Dataset A " << _numRecomputedEmbeddings << "created " << _refinedDatasetsA.back()->getNumPoints() << " points " << _refinedDatasetsA.back()->getNumDimensions() << " dimensions";

    // add a new dataset for the corresponding 1D embedding of the refined 2D embedding B
    QString dataset1DName = "1D Embedding " + refineEmbedding->getGuiName();
    _embedding1DRefinedDatasets.push_back(mv::data().createDerivedDataset(dataset1DName, refineEmbedding, refineEmbedding));
    auto& embedding1DRefined = _embedding1DRefinedDatasets.back();

    std::vector<float> initialData1DRefine;
    initialData1DRefine.resize(refineEmbedding->getNumPoints());

    embedding1DRefined->setData(initialData1DRefine.data(), refineEmbedding->getNumPoints(), 1);
    events().notifyDatasetAdded(embedding1DRefined);
    events().notifyDatasetDataChanged(embedding1DRefined);
    qDebug() << "onRefineFinished(): 1D Embedding for refined B created " << embedding1DRefined->getGuiName();


    TsneAnalysis* tsneAnalysis1D = new TsneAnalysis();

    //TsneSettingsAction* tsneSettingsAction1D = new TsneSettingsAction(this, refineEmbedding->getNumPoints());
    auto tsneSettingsAction1D = std::make_unique<TsneSettingsAction>(this, refineEmbedding->getNumPoints());
    tsneSettingsAction1D->getGeneralTsneSettingsAction().getNumDimensionOutputAction().setCurrentIndex(0); // set 1D output
    tsneSettingsAction1D->getGeneralTsneSettingsAction().getNumIterationsAction().setValue(500); // temp test

    mv::Task& datasetTask = embedding1DRefined->getTask();
    datasetTask.setName("t-SNE computation");
    datasetTask.setConfigurationFlag(Task::ConfigurationFlag::OverrideAggregateStatus);
    tsneAnalysis1D->setTask(&datasetTask);
    
    _refinedEmbedding2D = convertDatasetToEmbedding(refineEmbedding, 2);

    connect(tsneAnalysis1D, &TsneAnalysis::finished, tsneAnalysis1D, [this, tsneAnalysis1D]() {
        qDebug() << "1D tSNE for refined embedding B finished";
        tsneAnalysis1D->deleteLater(); 
        compute2DEmbeddingAWhenDrillInB(); // compute the recomputed 2D embedding A based on the refined subset 
        });

    connect(tsneAnalysis1D, &TsneAnalysis::embeddingUpdate, this, [this](const TsneData& tsneData) {
        auto& embedding1DRefined = _embedding1DRefinedDatasets.back();
        embedding1DRefined->setData(tsneData.getData().data(), tsneData.getNumPoints(), 1);
        events().notifyDatasetDataChanged(embedding1DRefined);
        });

    // Prepare data for computation
    qDebug() << "Prepare to compute 1D tsne for the refined embedding B";
    std::vector<float> data;
    std::vector<unsigned int> indices;

    auto& refinedDataset = refineEmbedding; // TODO: remove this, just use refineEmbedding
    std::vector<bool> enabledDimensions = refinedDataset->getDimensionsPickerAction().getEnabledDimensions();
    const auto numEnabledDimensions = std::count(enabledDimensions.begin(), enabledDimensions.end(), true);

    const auto numPoints = refinedDataset->isFull() ? refinedDataset->getNumPoints() : refinedDataset->indices.size();
    data.resize(numPoints * numEnabledDimensions);

    for (int i = 0; i < refinedDataset->getNumDimensions(); i++)
        if (enabledDimensions[i])
            indices.push_back(i);

    refinedDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, indices);
    qDebug() << "1D tSNE data prepared for 1D refined B";

    // Initialize embedding (1D)
    //auto initEmbedding1D = tsneSettingsAction1D->getInitalEmbeddingSettingsAction().getInitEmbedding(numPoints);
    std::vector<float> initEmbedding1D(refinedDataset->getNumPoints());
    refinedDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding1D, { 1 });

    // generate master embedding
    auto embedding2DB = convertDatasetToEmbedding(refineEmbedding, 2);

    tsneAnalysis1D->startComputation(
    tsneSettingsAction1D->getTsneParameters(),
    tsneSettingsAction1D->getKnnParameters(),
    std::move(data),
    numEnabledDimensions,
    &initEmbedding1D,
    &embedding2DB.getContainer()
    ); 

}

void DualAnalysisPlugin::compute2DEmbeddingAWhenDrillInB()
{
    qDebug() << "DualAnalysisPlugin::compute2DEmbeddingAWhenDrillInB()";

    auto& refinedDatasetA = _refinedDatasetsA.back();
    qDebug() << "compute2DEmbeddingAWhenDrillInB(): Refined Dataset A: " << refinedDatasetA->getGuiName();

    // add a new dataset for the recomputed 2D embedding A
    _recomputedEmbedding2DDatasetsA.push_back(mv::data().createDerivedDataset<Points>(QString("Recomputed 2D Embedding A %1").arg(_numRecomputedEmbeddings), _refinedDatasetsA.back(), _refinedDatasetsA.back()));
    auto& recomputed2DEmbeddingA = _recomputedEmbedding2DDatasetsA.back();
    recomputed2DEmbeddingA->setData(_embedding2DA.getContainer().data(), _datasetA->getNumPoints(), 2); // start with the previous 2D embedding A
    events().notifyDatasetAdded(recomputed2DEmbeddingA);
    events().notifyDatasetDataChanged(recomputed2DEmbeddingA);
    qDebug() << "Recomputed 2D Embedding A " << _numRecomputedEmbeddings << "created";

    TsneAnalysis* tsneAnalysis = new TsneAnalysis();

    //auto tsneSettingsAction = new TsneSettingsAction(this, refinedDatasetA->getNumPoints());
    auto tsneSettingsAction = std::make_unique<TsneSettingsAction>(this, refinedDatasetA->getNumPoints());
    tsneSettingsAction->getGeneralTsneSettingsAction().getNumIterationsAction().setValue(500);
    tsneSettingsAction->getKnnParameters().setKnnAlgorithm(hdi::dr::knn_library::KNN_HNSW);

    mv::Task& datasetTask = recomputed2DEmbeddingA->getTask();
    datasetTask.setName("t-SNE computation");
    datasetTask.setConfigurationFlag(Task::ConfigurationFlag::OverrideAggregateStatus);
    tsneAnalysis->setTask(&datasetTask);

    connect(tsneAnalysis, &TsneAnalysis::finished, tsneAnalysis, [this, tsneAnalysis]() {
        tsneAnalysis->deleteLater(); 
        compute1DEmbeddingAWhenDrillInB();
        });

    connect(tsneAnalysis, &TsneAnalysis::embeddingUpdate, this, [this, &recomputed2DEmbeddingA](const TsneData& tsneData) {
        recomputed2DEmbeddingA->setData(tsneData.getData().data(), tsneData.getNumPoints(), 2);
        events().notifyDatasetDataChanged(recomputed2DEmbeddingA);
        });

   
    // prepare data
    std::vector<float> data;
    std::vector<unsigned int> indices;
  
    std::vector<bool> enabledDimensions = refinedDatasetA->getDimensionsPickerAction().getEnabledDimensions();
    const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });
    const auto numPoints = refinedDatasetA->isFull() ? refinedDatasetA->getNumPoints() : refinedDatasetA->indices.size();
    data.resize(numPoints * numEnabledDimensions);

    for (int i = 0; i < refinedDatasetA->getNumDimensions(); i++)
        if (enabledDimensions[i])
            indices.push_back(i);

    refinedDatasetA->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, indices);

    // Init embedding: random or set from other dataset, e.g. PCA
    auto initEmbedding = tsneSettingsAction->getInitalEmbeddingSettingsAction().getInitEmbedding(numPoints); // random initialization

    qDebug() << "Start 2D tSNE for the recomputed 2D embedding A";

    // recomputed 2D embedding A
    tsneAnalysis->startComputation(
        tsneSettingsAction->getTsneParameters(),
        tsneSettingsAction->getKnnParameters(),
        std::move(data),
        numEnabledDimensions,
        &initEmbedding);
}

void DualAnalysisPlugin::compute1DEmbeddingAWhenDrillInB()
{
    qDebug() << "DualAnalysisPlugin::compute1DEmbeddingAWhenDrillInB()";

    auto& recomputed2DEmbeddingA = _recomputedEmbedding2DDatasetsA.back();
    qDebug() << "compute1DEmbeddingAWhenDrillInB(): Recomputed 2D Embedding A: " << recomputed2DEmbeddingA->getGuiName();

    // add a new dataset for the recomputed 1D embedding A
    _recomputedEmbedding1DDatasetsA.push_back(mv::data().createDerivedDataset<Points>(QString("Recomputed 1D Embedding A %1").arg(_numRecomputedEmbeddings), recomputed2DEmbeddingA, recomputed2DEmbeddingA));
    auto& recomputed1DEmbeddingA = _recomputedEmbedding1DDatasetsA.back();

    std::vector<float> initialData1DRecomputed;
    initialData1DRecomputed.resize(recomputed2DEmbeddingA->getNumPoints());

    recomputed1DEmbeddingA->setData(initialData1DRecomputed.data(), recomputed2DEmbeddingA->getNumPoints(), 1);
    events().notifyDatasetAdded(recomputed1DEmbeddingA);
    events().notifyDatasetDataChanged(recomputed1DEmbeddingA);
    qDebug() << "Recomputed 1D Embedding A " << _numRecomputedEmbeddings << "created";

    TsneAnalysis* tsneAnalysis1D = new TsneAnalysis();
    //TsneSettingsAction* tsneSettingsAction1D = new TsneSettingsAction(this, recomputed2DEmbeddingA->getNumPoints());
    auto tsneSettingsAction1D = std::make_unique<TsneSettingsAction>(this, recomputed2DEmbeddingA->getNumPoints());
    tsneSettingsAction1D->getGeneralTsneSettingsAction().getNumDimensionOutputAction().setCurrentIndex(0); // set 1D output
    tsneSettingsAction1D->getGeneralTsneSettingsAction().getNumIterationsAction().setValue(500);

    mv::Task& datasetTask = recomputed1DEmbeddingA->getTask();
    datasetTask.setName("t-SNE computation");
    datasetTask.setConfigurationFlag(Task::ConfigurationFlag::OverrideAggregateStatus);
    tsneAnalysis1D->setTask(&datasetTask);

    connect(tsneAnalysis1D, &TsneAnalysis::finished, tsneAnalysis1D, [this, tsneAnalysis1D]() {
        qDebug() << "1D tSNE for recomputed embedding A finished";
        tsneAnalysis1D->deleteLater();
        });

    connect(tsneAnalysis1D, &TsneAnalysis::embeddingUpdate, this, [this, &recomputed1DEmbeddingA](const TsneData& tsneData) {
        recomputed1DEmbeddingA->setData(tsneData.getData().data(), tsneData.getNumPoints(), 1);
        events().notifyDatasetDataChanged(recomputed1DEmbeddingA);
        });


    // Prepare data for computation
    std::vector<float> data;
    std::vector<unsigned int> indices;

    std::vector<bool> enabledDimensions = recomputed2DEmbeddingA->getDimensionsPickerAction().getEnabledDimensions();
    const auto numEnabledDimensions = std::count(enabledDimensions.begin(), enabledDimensions.end(), true);

    const auto numPoints = recomputed2DEmbeddingA->isFull() ? recomputed2DEmbeddingA->getNumPoints() : recomputed2DEmbeddingA->indices.size();
    data.resize(numPoints * numEnabledDimensions);

    for (int i = 0; i < recomputed2DEmbeddingA->getNumDimensions(); i++)
        if (enabledDimensions[i])
            indices.push_back(i);

    recomputed2DEmbeddingA->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, indices);
    qDebug() << "1D tSNE data prepared for 1D recomputedA: numPoints=" << numPoints << " numEnabledDimensions=" << numEnabledDimensions;

    // Initialize embedding (1D)
    //auto initEmbedding1D = tsneSettingsAction1D->getInitalEmbeddingSettingsAction().getInitEmbedding(numPoints);
    std::vector<float> initEmbedding1D(recomputed1DEmbeddingA->getNumPoints() * recomputed1DEmbeddingA->getNumDimensions());
    recomputed2DEmbeddingA->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding1D, { 1 });

    // master embedding
    auto masterEmbedding = convertDatasetToEmbedding(recomputed2DEmbeddingA, 2);
    qDebug() << "Master embedding created for 1D recomputedA " << masterEmbedding.numDataPoints() << " points " << masterEmbedding.numDimensions() << " dimensions";

    tsneAnalysis1D->startComputation(
        tsneSettingsAction1D->getTsneParameters(),
        tsneSettingsAction1D->getKnnParameters(),
        std::move(data),
        numEnabledDimensions,
        &initEmbedding1D,
        &masterEmbedding.getContainer()
    );

}

void DualAnalysisPlugin::onAlignmentTriggered()
{
    qDebug() << "DualAnalysisPlugin::onAlignmentTriggered";

    // compute connections between 1D embeddings - connectionsAB 
    
    _connectionsAB.resize(_embedding1DDatasetA->getNumPoints());

    auto embeddingSourceDatasetA = _embedding2DDatasetA->getSourceDataset<Points>();
    auto embeddingSourceDatasetB = _embedding2DDatasetB->getSourceDataset<Points>();// for hsne embedding this is the hsne selection helper, a subset of the original source dataset
    qDebug() << "embeddingSourceDatasetA: " << embeddingSourceDatasetA->getGuiName() << " embeddingSourceDatasetB: " << embeddingSourceDatasetB->getGuiName();
    qDebug() << "embedding2DDataset->getSourceDataset->getSourceDataset" << _embedding2DDatasetA->getSourceDataset<Points>()->getSourceDataset<Points>()->getGuiName();

    int numDimensions = embeddingSourceDatasetA->getNumPoints();
    int numDimensionsFull = embeddingSourceDatasetB->getNumDimensions();
    int numPoints = embeddingSourceDatasetB->getSourceDataset<Points>()->getNumPoints();
    int numPointsLocal = _embedding2DDatasetB->getNumPoints(); // num of points in the embedding B
    qDebug() << "numDimensions: " << numDimensions << " numDimensionsFull: " << numDimensionsFull << " numPoints: " << numPoints;
    qDebug() << "numPointsLocal: " << numPointsLocal << "numPoints source of sourceB " << embeddingSourceDatasetB->getSourceDataset<Points>()->getNumPoints();

    auto start1 = std::chrono::high_resolution_clock::now();
    // Get the global indices of embedding A (dimension embedding)
    std::vector<std::uint32_t> localGlobalIndicesA;
    embeddingSourceDatasetA->getGlobalIndices(localGlobalIndicesA);

    // Compute minimum and range for each dimension
    std::vector<float> columnMins(numDimensions, std::numeric_limits<float>::max());
    std::vector<float> columnRanges(numDimensions, 0.0f);

#pragma omp parallel for
    for (int dimLocalIdx = 0; dimLocalIdx < numDimensions; dimLocalIdx++) {
        int dimGlobalIdx = static_cast<int>(localGlobalIndicesA[dimLocalIdx]);

        float minValue = std::numeric_limits<float>::max();
        float maxValue = -std::numeric_limits<float>::max();

        for (int i = 0; i < numPoints; i++) {
            float val = embeddingSourceDatasetB->getValueAt(i * numDimensionsFull + dimGlobalIdx);
            if (val < minValue) minValue = val;
            if (val > maxValue) maxValue = val;
        }

        columnMins[dimLocalIdx] = minValue;
        columnRanges[dimLocalIdx] = maxValue - minValue;
    }

    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    qDebug() << "Time to compute columnMins and columnRanges: " << duration1.count() << " ms";

    auto start2 = std::chrono::high_resolution_clock::now();
    // Iterate over dataset B and compute connections to dataset A
    // WorkInProgress: try to fix with landmarks


    std::vector<std::uint32_t> localGlobalIndicesB;
    embeddingSourceDatasetB->getGlobalIndices(localGlobalIndicesB);

#pragma omp parallel for
    for (int dimLocalIdx = 0; dimLocalIdx < numDimensions; dimLocalIdx++) {
        float threshold = columnMins[dimLocalIdx] + 0.9f * columnRanges[dimLocalIdx];
        int dimGlobalIdx = static_cast<int>(localGlobalIndicesA[dimLocalIdx]);

        // Each thread deals with a different dimension => no concurrency on the same _connectionsAB index
        for (int cellLocalIndex = 0; cellLocalIndex < numPoints; cellLocalIndex++) {
            int cellGlobalIndex = static_cast<int>(localGlobalIndicesB[cellLocalIndex]);
            float expression = embeddingSourceDatasetB->getValueAt(cellGlobalIndex * numDimensionsFull + dimGlobalIdx);

            if (expression > threshold) 
            {
                _connectionsAB[dimLocalIdx][cellLocalIndex]++;
            }
        }
    }

    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    qDebug() << "Time to compute connectionsAB: " << duration2.count() << " ms";

    qDebug() << "ConnectionsAB computed" << _connectionsAB.size();
    qDebug() << "ConnectionsAB[0] size " << _connectionsAB[0].size();
    qDebug() << "ConnectionsAB[0][0] " << _connectionsAB[0][0] << " ConnectionsAB[0][1] " << _connectionsAB[0][1] << " ConnectionsAB[0][2] " << _connectionsAB[0][2];


    std::vector<float> test1(_embedding1DDatasetB->getNumPoints());
    _embedding1DDatasetB->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(test1, { 0 });
    qDebug() << "test 1 [0] " << test1[0] << " [1] " << test1[1] << " [2] " << test1[2];

    std::vector<float> test3;
    _embedding1DDatasetB->extractDataForDimension(test3, 0);
    qDebug() << "test 3 [0] " << test3[0] << " [1] " << test3[1] << " [2] " << test3[2];

    qDebug() << "test1 size " << test1.size() << " test3 size " << test3.size();

    qDebug() << "embedding1DDatasetB numPoints = " << _embedding1DDatasetB->getNumPoints() << " dim = " << _embedding1DDatasetB->getNumDimensions();

    _embedding1DB = convertDatasetToEmbedding(_embedding1DDatasetB, 1);

    std::vector<float> test4 = _embedding1DB.getContainer();
    qDebug() << "test4 size " << test4.size() << "test 4 [0] " << test4[0] << " [1] " << test4[1] << " [2] " << test4[2];


    // align 1D tSNE A using 1D tSNE B as master embedding - equalizer with connections
    //startAlignmentComputation(_embedding2DDatasetA, _embedding1DDatasetA, _1DtsneAnalysisA, _1DtsneSettingsActionA, _1DdataPreparationTaskA, _embedding1DB); // 1DB-1DA + connections

    // test use new tsne analysis instance to align 1DA to 1DB - start ------------------------------------------------------------------------------
    TsneAnalysis* tsneAnalysis1D = new TsneAnalysis();
    auto tsneSettingsAction1D = std::make_unique<TsneSettingsAction>(this, _embedding1DDatasetA->getNumPoints());
    tsneSettingsAction1D->getGeneralTsneSettingsAction().getNumDimensionOutputAction().setCurrentIndex(0); // set 1D output
    tsneSettingsAction1D->getGeneralTsneSettingsAction().getNumIterationsAction().setValue(500); // temp test

    mv::Task& datasetTask = _embedding1DDatasetA->getTask();
    datasetTask.setName("t-SNE computation");
    datasetTask.setConfigurationFlag(Task::ConfigurationFlag::OverrideAggregateStatus);
    tsneAnalysis1D->setTask(&datasetTask);

    connect(tsneAnalysis1D, &TsneAnalysis::finished, tsneAnalysis1D, [this, tsneAnalysis1D]() {
        qDebug() << "1D tSNE A aligned to 1D tSNE B finished";
        tsneAnalysis1D->deleteLater();
        qDebug() << "1D tSNE A aligned to 1D tSNE B deleted";
        align2DAto1DA();
        });

    connect(tsneAnalysis1D, &TsneAnalysis::embeddingUpdate, this, [this](const TsneData& tsneData) {
        _embedding1DDatasetA->setData(tsneData.getData().data(), tsneData.getNumPoints(), 1);
        events().notifyDatasetDataChanged(_embedding1DDatasetA);
        });

    // Prepare data for computation
    std::vector<float> data;
    std::vector<unsigned int> indices;

    std::vector<bool> enabledDimensions = _embedding2DDatasetA->getDimensionsPickerAction().getEnabledDimensions();
    const auto numEnabledDimensions = std::count(enabledDimensions.begin(), enabledDimensions.end(), true);

    const auto numPointsA = _embedding2DDatasetA->isFull() ? _embedding2DDatasetA->getNumPoints() : _embedding2DDatasetA->indices.size();
    data.resize(numPointsA * numEnabledDimensions);

    for (int i = 0; i < _embedding2DDatasetA->getNumDimensions(); i++)
        if (enabledDimensions[i])
            indices.push_back(i);

    _embedding2DDatasetA->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, indices);
    qDebug() << "1D tSNE data prepared for align 1DA to 1DB ";

    // Initialize embedding (1D)
    std::vector<float> initEmbedding1D(_embedding1DDatasetA->getNumPoints());
    _embedding1DDatasetA->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding1D, { 0 });

    // generate master embedding
    auto embedding1DB = convertDatasetToEmbedding(_embedding1DDatasetB, 1);

    tsneAnalysis1D->startComputation(
        tsneSettingsAction1D->getTsneParameters(),
        tsneSettingsAction1D->getKnnParameters(),
        std::move(data),
        numEnabledDimensions,
        &initEmbedding1D,
        &embedding1DB.getContainer(),
        _connectionsAB
    );


    // test use new tsne analysis instance 1DA to 1DB - end ------------------------------------------------------------------------------



    // align 2D tsne A using 1D tsne A as master embedding - TODO: need to wait until the above computation is finished

 //   _triggerAlignment = 0; // reset to 0 when align button is hit
 //   connect(&_1DtsneAnalysisA, &TsneAnalysis::finished, this, [this]() {
 //       qDebug() << "DualAnalysisPlugin::onAlignmentTriggered: 1D tSNE A alignment finished";
 //       if (_triggerAlignment == 1) {
 //           qDebug() << "DualAnalysisPlugin::onAlignmentTriggered: 1D tSNE A alignment already finished";
 //           disconnect(&_1DtsneAnalysisA, &TsneAnalysis::finished, nullptr, nullptr);
 //           qDebug() << "DualAnalysisPlugin::onAlignmentTriggered: disconnect 1D tSNE A";
 //           return;
 //       }

 //       _embedding1DA = convertDatasetToEmbedding(_embedding1DDatasetA, 1);
 //       startAlignmentComputation(_datasetA, _embedding2DDatasetA, _tsneAnalysisA, _tsneSettingsActionA, _dataPreparationTaskA, _embedding1DA);// 1DA-2DA
 //       _triggerAlignment = 1; 

	//});

    // FIXME: need to disconnect this? Otherwise, would be still linked next time 1D tSNE is computed - add a stop alignment button

}

void DualAnalysisPlugin::align2DAto1DA()
{
    // test use new tsne analysis instance 2DA to 1DA
	qDebug() << "DualAnalysisPlugin::align2DAto1DA()";

	// align 2D tsne A using 1D tsne A as master embedding
    TsneAnalysis* tsneAnalysis = new TsneAnalysis();

    auto tsneSettingsAction = std::make_unique<TsneSettingsAction>(this, _embedding2DDatasetA->getNumPoints());
    tsneSettingsAction->getGeneralTsneSettingsAction().getNumIterationsAction().setValue(1000);//1000
    tsneSettingsAction->getGeneralTsneSettingsAction().getPerplexityAction().setValue(5); // temp test
    tsneSettingsAction->getKnnParameters().setKnnAlgorithm(hdi::dr::knn_library::KNN_HNSW);

    mv::Task& datasetTask = _embedding2DDatasetA->getTask();
    datasetTask.setName("t-SNE computation");
    datasetTask.setConfigurationFlag(Task::ConfigurationFlag::OverrideAggregateStatus);
    tsneAnalysis->setTask(&datasetTask);

    connect(tsneAnalysis, &TsneAnalysis::finished, tsneAnalysis, [this, tsneAnalysis]() {
        tsneAnalysis->deleteLater();
        });

    connect(tsneAnalysis, &TsneAnalysis::embeddingUpdate, this, [this](const TsneData& tsneData) {
        _embedding2DDatasetA->setData(tsneData.getData().data(), tsneData.getNumPoints(), 2);
        events().notifyDatasetDataChanged(_embedding2DDatasetA);
        });


    // prepare data
    std::vector<float> data;
    std::vector<unsigned int> indices;

    std::vector<bool> enabledDimensions = _datasetA->getDimensionsPickerAction().getEnabledDimensions();
    const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });
    const auto numPoints = _datasetA->isFull() ? _datasetA->getNumPoints() : _datasetA->indices.size();
    data.resize(numPoints * numEnabledDimensions);

    for (int i = 0; i < _datasetA->getNumDimensions(); i++)
        if (enabledDimensions[i])
            indices.push_back(i);

    _datasetA->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, indices);

    // Init embedding
    std::vector<float> initEmbedding2D(2 * _embedding2DDatasetA->getNumPoints());
    _embedding2DDatasetA->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding2D, { 0, 1 });

    //test 06.01
    // fill in the y coorinates using the 1D embedding A
    std::vector<float> initEmbedding2Dy(_embedding2DDatasetA->getNumPoints());
    _embedding1DDatasetA->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding2Dy, { 0 });
    for (int i = 0; i < _embedding2DDatasetA->getNumPoints(); i++)
    {
        initEmbedding2D[2 * i + 1] = initEmbedding2Dy[i];
    }
    qDebug() << "initEmbedding2D size " << initEmbedding2D.size() << " initEmbedding2Dy size " << initEmbedding2Dy.size();

    //auto test = tsneSettingsAction->getGradientDescentSettingsAction().getExaggerationIterAction().getValue();
    //qDebug() << "exaggeration iter " << test;

    // test to turn off exaggeration
    //tsneSettingsAction->getGradientDescentSettingsAction().getExaggerationIterAction().setValue(0);
    //tsneSettingsAction->getGradientDescentSettingsAction().getExponentialDecayAction().setValue(0);
    //test = tsneSettingsAction->getGradientDescentSettingsAction().getExaggerationIterAction().getValue();
    //qDebug() << "exaggeration iter " << test;


    // generate master embedding
    auto embedding1DA = convertDatasetToEmbedding(_embedding1DDatasetA, 1);
    auto embedding1DB = convertDatasetToEmbedding(_embedding1DDatasetB, 1);

    qDebug() << "Start 2D tSNE to align 2DA to 1DA";

    // recomputed 2D embedding A
    tsneAnalysis->startComputation(
        tsneSettingsAction->getTsneParameters(),
        tsneSettingsAction->getKnnParameters(),
        std::move(data),
        numEnabledDimensions,
        &initEmbedding2D,
        & embedding1DA.getContainer()
    );

    // test to align 2DA to 1DB with connections
    /*tsneAnalysis->startComputation(
		tsneSettingsAction->getTsneParameters(),
		tsneSettingsAction->getKnnParameters(),
		std::move(data),
		numEnabledDimensions,
		&initEmbedding2D,
		&embedding1DB.getContainer(),
		_connectionsAB
	);*/



}

void DualAnalysisPlugin::startAlignmentComputation(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction, mv::Task& dataPreparationTask,
    hdi::data::Embedding<float>& masterEmbedding)
{
    embeddingDataset->getTask().setRunning();

    dataPreparationTask.setEnabled(true);
    dataPreparationTask.setRunning();

    // Create list of data from the enabled dimensions
    std::vector<float> data;
    std::vector<unsigned int> indices;

    // Extract the enabled dimensions from the data
    std::vector<bool> enabledDimensions = inputDataset->getDimensionsPickerAction().getEnabledDimensions();

    const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    tsneSettingsAction->getGeneralTsneSettingsAction().getNumberOfComputatedIterationsAction().reset();

    const auto numPoints = inputDataset->isFull() ? inputDataset->getNumPoints() : inputDataset->indices.size();
    data.resize(numPoints * numEnabledDimensions);

    for (int i = 0; i < inputDataset->getNumDimensions(); i++)
        if (enabledDimensions[i])
            indices.push_back(i);

    inputDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, indices);

    tsneSettingsAction->getComputationAction().getRunningAction().setChecked(true);

    // test to add connection here 27.12
    connect(&tsneAnalysis, &TsneAnalysis::embeddingUpdate, this, [this, &tsneAnalysis, &tsneSettingsAction, &embeddingDataset](const TsneData tsneData) {
        qDebug() << "DualAnalysisPlugin::startAlignmentComputation: embeddingUpdate";
        // Update the output points dataset with new data from the TSNE analysis
        embeddingDataset->setData(tsneData.getData().data(), tsneData.getNumPoints(), tsneSettingsAction->getGeneralTsneSettingsAction().getNumDimensionOutputAction().getCurrentText().toInt());

        tsneSettingsAction->getGeneralTsneSettingsAction().getNumberOfComputatedIterationsAction().setValue(tsneAnalysis.getNumIterations() - 1);

        // Notify others that the embedding data changed
        events().notifyDatasetDataChanged(embeddingDataset);
        });


	if (embeddingDataset->getNumDimensions() == 1) // if slave is 1D, assume it is 1DB-1DA with connections
	{
        qDebug() << "DualAnalysisPlugin::startAlignmentComputation: 1DB-1DA with connections";

		std::vector<float> initEmbedding(numPoints);
		embeddingDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding, { 0 });

        // test with a initial embedding with all zeros 
        //std::vector<float> initEmbedding(numPoints, 0.0f);

        // test with a initial embedding with input dataset y axis
        /*std::vector<float> initEmbedding(numPoints);
        inputDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding, { 1 });*/

		dataPreparationTask.setFinished();

		tsneAnalysis.startComputation(tsneSettingsAction->getTsneParameters(), tsneSettingsAction->getKnnParameters(), std::move(data), numEnabledDimensions, &initEmbedding, &masterEmbedding.getContainer(), _connectionsAB);
	}
    else if (embeddingDataset->getNumDimensions() == 2) // if slave is 2D, assume it is 1DA-2DA
    {
        qDebug() << "DualAnalysisPlugin::startAlignmentComputation: 1DA-2DA";
		std::vector<float> initEmbedding(2 * numPoints);
		embeddingDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding, { 0, 1 });

		dataPreparationTask.setFinished();

		tsneAnalysis.startComputation(tsneSettingsAction->getTsneParameters(), tsneSettingsAction->getKnnParameters(), std::move(data), numEnabledDimensions, &initEmbedding, &masterEmbedding.getContainer());
	}


    // test to see if the above code works with initial embedding
    //embeddingDataset->getTask().setRunning();

    //dataPreparationTask.setEnabled(false);

    //// Create list of data from the enabled dimensions
    //std::vector<float> data;
    //std::vector<unsigned int> indices;

    //// Extract the enabled dimensions from the data
    //std::vector<bool> enabledDimensions = inputDataset->getDimensionsPickerAction().getEnabledDimensions();

    //const auto numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    //tsneSettingsAction->getGeneralTsneSettingsAction().getNumberOfComputatedIterationsAction().reset();

    //const auto numPoints = inputDataset->isFull() ? inputDataset->getNumPoints() : inputDataset->indices.size();
    //data.resize(numPoints * numEnabledDimensions);

    //for (int i = 0; i < inputDataset->getNumDimensions(); i++)
    //    if (enabledDimensions[i])
    //        indices.push_back(i);

    //inputDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(data, indices);

    //tsneSettingsAction->getComputationAction().getRunningAction().setChecked(true);

    //if (embeddingDataset->getNumDimensions() == 1) // if slave is 1D, assume it is 1DB-1DA with connections
    //{
    //    std::vector<float> initEmbedding(numPoints);
    //    embeddingDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding, { 0 });
    //    tsneAnalysis.startComputation(tsneSettingsAction->getTsneParameters(), tsneSettingsAction->getKnnParameters(), std::move(data), numEnabledDimensions, &initEmbedding, &masterEmbedding.getContainer(), _connectionsAB);
    //}
    //else if (embeddingDataset->getNumDimensions() == 2) // if slave is 2D, assume it is 1DA-2DA
    //{
    //    std::vector<float> initEmbedding(2 * numPoints);
    //    embeddingDataset->populateDataForDimensions<std::vector<float>, std::vector<unsigned int>>(initEmbedding, { 0, 1 });
    //    tsneAnalysis.startComputation(tsneSettingsAction->getTsneParameters(), tsneSettingsAction->getKnnParameters(), std::move(data), numEnabledDimensions, &initEmbedding, &masterEmbedding.getContainer());
    //}
}


/******************************************************************************
 * Serialization
 ******************************************************************************/

void DualAnalysisPlugin::fromVariantMap(const QVariantMap& variantMap)
{
    _loadingFromProject = true;

    AnalysisPlugin::fromVariantMap(variantMap);

    /*_settingsAction.fromParentVariantMap(variantMap);
    qDebug() << "DualAnalysisPlugin::fromVariantMap: settingsAction";*/

    /*--------------------------------------------------------------
      datasets
    *--------------------------------------------------------------*/
    _datasetA = mv::data().getDataset(variantMap["transposedDataGuid"].toString());
    _embedding2DDatasetA = mv::data().getDataset(variantMap["embedding2DDatasetAGuid"].toString());
    _embedding1DDatasetA = mv::data().getDataset(variantMap["embedding1DDatasetAGuid"].toString());
    _embedding2DDatasetB = mv::data().getDataset(variantMap["embedding2DDatasetBGuid"].toString());
    _embedding1DDatasetB = mv::data().getDataset(variantMap["embedding1DDatasetBGuid"].toString());
    qDebug() << "DualAnalysisPlugin::fromVariantMap: datasets";

	/*--------------------------------------------------------------
	  embedding A settings
	*--------------------------------------------------------------*/
    initializeEmbeddingA();

	variantMapMustContain(variantMap, "TSNE Settings A");
	_tsneSettingsActionA->fromVariantMap(variantMap["TSNE Settings A"].toMap());

    if (_tsneSettingsActionA->getGeneralTsneSettingsAction().getSaveProbDistAction().isChecked())
    {
        if (variantMap.contains("probabilityDistribution A"))
        {
            //qDebug() << "DualAnalysisPlugin::fromVariantMap variantMap.containsprobabilityDistribution A";
             
            const auto loadPathHierarchy = QDir::cleanPath(projects().getTemporaryDirPath(AbstractProjectManager::TemporaryDirType::Open) + QDir::separator() + variantMap["probabilityDistribution A"].toString());

            std::ifstream loadFile(loadPathHierarchy.toStdString().c_str(), std::ios::in | std::ios::binary);

            if (loadFile.is_open())
            {
                hdi::data::IO::loadSparseMatrix(_probDistMatrixA, loadFile, nullptr);

                _tsneSettingsActionA->getComputationAction().getContinueComputationAction().setEnabled(true);
            }
            else
                qWarning("TsneAnalysisPlugin::fromVariantMap: t-SNE probability distribution A was NOT loaded successfully");
        }
        else
            qWarning("TsneAnalysisPlugin::fromVariantMap: t-SNE probability distribution A cannot be loaded from project since the project file does not seem to contain a corresponding file.");
    }

    _embedding2DDatasetA->_infoAction->collapse();

    qDebug() << "DualAnalysisPlugin::fromVariantMap: embedding A settings";

    /*--------------------------------------------------------------
      embedding B settings
    *--------------------------------------------------------------*/
    _settingsAction.fromParentVariantMap(variantMap); // settingsAction is only for B
    _embedding2DDatasetB->addAction(_settingsAction);
       
    initializeEmbeddingB();

    if (variantMap.contains("TSNE Settings B"))
    {
        qDebug() << "DualAnalysisPlugin::fromVariantMap variantMap.contains TSNE Settings B";

        _settingsAction.getEmbeddingAlgorithmAction().setCurrentIndex(1); // t-SNE
        _settingsAction.getEmbeddingAlgorithmAction().setEnabled(false);

		_tsneSettingsActionB = new TsneSettingsAction(this, _datasetB->getNumPoints());
		_tsneSettingsActionB->fromVariantMap(variantMap["TSNE Settings B"].toMap());		

		setup2DTsneForDataset(_datasetB, _embedding2DDatasetB, _tsneAnalysisB, _tsneSettingsActionB, _probDistMatrixB, _dataPreparationTaskB);

        if (_tsneSettingsActionB->getGeneralTsneSettingsAction().getSaveProbDistAction().isChecked())
        {
            if (variantMap.contains("probabilityDistribution B"))
            {
                //qDebug() << "DualAnalysisPlugin::fromVariantMap variantMap.contains probabilityDistribution B";
                
                // Load HSNE Hierarchy
                const auto loadPathHierarchy = QDir::cleanPath(projects().getTemporaryDirPath(AbstractProjectManager::TemporaryDirType::Open) + QDir::separator() + variantMap["probabilityDistribution B"].toString());

                std::ifstream loadFile(loadPathHierarchy.toStdString().c_str(), std::ios::in | std::ios::binary);

                if (loadFile.is_open())
                {
                    hdi::data::IO::loadSparseMatrix(_probDistMatrixB, loadFile, nullptr);

                    _tsneSettingsActionB->getComputationAction().getContinueComputationAction().setEnabled(true);
                }
                else
                    qWarning("TsneAnalysisPlugin::fromVariantMap: t-SNE probability distribution B was NOT loaded successfully");
            }
            else
                qWarning("TsneAnalysisPlugin::fromVariantMap: t-SNE probability distribution B cannot be loaded from project since the project file does not seem to contain a corresponding file.");
        }
    }
    else if (variantMap.contains("HSNE Settings B"))
    {
        qDebug() << "DualAnalysisPlugin::fromVariantMap variantMap.contains HSNE Settings B";

        _settingsAction.getEmbeddingAlgorithmAction().setCurrentIndex(2); // HSNE
        _settingsAction.getEmbeddingAlgorithmAction().setEnabled(false);

        _hsneSettingsAction = new HsneSettingsAction(this);
		_hsneSettingsAction->fromVariantMap(variantMap["HSNE Settings B"].toMap());

        setupHSNEForDataset(_datasetB, _embedding2DDatasetB, _tsneAnalysisHSNEB);

        std::vector<bool> enabledDimensions = _datasetB->getDimensionsPickerAction().getEnabledDimensions();
        _hierarchy->setDataAndParameters(_datasetB, _embedding2DDatasetB, _hsneSettingsAction->getHsneParameters(), _hsneSettingsAction->getKnnParameters(), std::move(enabledDimensions));

        auto& hsne = _hierarchy->getHsne();
        hsne.setDimensionality(_hierarchy->getNumDimensions());

        if (_hsneSettingsAction->getHierarchyConstructionSettingsAction().getSaveHierarchyToProjectAction().isChecked())
        {
            //qDebug() << "DualAnalysisPlugin::fromVariantMap save hierarchy to project is checked B ";

            if (variantMap.contains("HsneHierarchy B") && variantMap.contains("HsneInfluenceHierarchy B"))
            {
                hdi::utils::CoutLog log;

                // Load HSNE Hierarchy
                const auto loadPathHierarchy = QDir::cleanPath(projects().getTemporaryDirPath(AbstractProjectManager::TemporaryDirType::Open) + QDir::separator() + variantMap["HsneHierarchy B"].toString());
                bool loadedHierarchy = _hierarchy->loadCacheHsneHierarchy(loadPathHierarchy.toStdString(), log);

                // Load HSNE InfluenceHierarchy
                const auto loadPathInfluenceHierarchy = QDir::cleanPath(projects().getTemporaryDirPath(AbstractProjectManager::TemporaryDirType::Open) + QDir::separator() + variantMap["HsneInfluenceHierarchy B"].toString());
                bool loadedInfluenceHierarchy = _hierarchy->loadCacheHsneInfluenceHierarchy(loadPathInfluenceHierarchy.toStdString(), _hierarchy->getInfluenceHierarchy().getMap());

                _hierarchy->setIsInitialized(true);

                if (!loadedHierarchy || !loadedInfluenceHierarchy)
                    qWarning("fromVariantMap: HSNE hierarchy B was NOT loaded successfully");
            }
            else
                qWarning("fromVariantMap: HSNE hierarchy B cannot be loaded from project since the project file does not seem to contain a saved HSNE hierarchy");
        }

        _selectionHelperData = mv::data().getDataset(variantMap["selectionHelperDataGUID B"].toString());
        _hsneSettingsAction->getGeneralHsneSettingsAction().getStartAction().setText("Recompute");
        _hsneSettingsAction->getGeneralHsneSettingsAction().getStartAction().setToolTip("Recomputing does not change the selection mapping.\n If the data size changed, prefer creating a new HSNE analysis.");
    }

    _embedding2DDatasetB->_infoAction->collapse();

    qDebug() << "DualAnalysisPlugin::fromVariantMap: embedding B settings";

    // TODO: check if something in init() is not yet serialized
    
    _loadingFromProject = false;
}

QVariantMap DualAnalysisPlugin::toVariantMap() const
{
    QVariantMap variantMap = AnalysisPlugin::toVariantMap();

    _settingsAction.insertIntoVariantMap(variantMap);

    /*--------------------------------------------------------------
      data ids
    *--------------------------------------------------------------*/
    variantMap.insert("transposedDataGuid", _datasetA.getDatasetId());
    variantMap.insert("embedding2DDatasetAGuid", _embedding2DDatasetA.getDatasetId());
    variantMap.insert("embedding1DDatasetAGuid", _embedding1DDatasetA.getDatasetId());
    variantMap.insert("embedding2DDatasetBGuid", _embedding2DDatasetB.getDatasetId());
    variantMap.insert("embedding1DDatasetBGuid", _embedding1DDatasetB.getDatasetId());

    
    /*--------------------------------------------------------------
      embedding A settings
    *--------------------------------------------------------------*/
    variantMap["TSNE Settings A"] = _tsneSettingsActionA->toVariantMap();

    const auto probabilityDistribution = _tsneAnalysisA.getProbabilityDistribution();

    if (_tsneSettingsActionA->getGeneralTsneSettingsAction().getSaveProbDistAction().isChecked() && probabilityDistribution != std::nullopt)
    {
        const auto fileName = QUuid::createUuid().toString(QUuid::WithoutBraces) + ".bin";
        const auto filePath = QDir::cleanPath(projects().getTemporaryDirPath(AbstractProjectManager::TemporaryDirType::Save) + QDir::separator() + fileName).toStdString();

        std::ofstream saveFile(filePath, std::ios::out | std::ios::binary);

        if (!saveFile.is_open())
            std::cerr << "Caching failed. File could not be opened. " << std::endl;
        else
        {
            hdi::data::IO::saveSparseMatrix(*probabilityDistribution.value(), saveFile, nullptr);
            saveFile.close();
            variantMap["probabilityDistribution A"] = fileName;
        }
    }

    /*--------------------------------------------------------------
     embedding B
    *--------------------------------------------------------------*/
    if (_tsneSettingsActionB != nullptr)
    {
        qDebug() << "toVariantMap() tsne was chosen for embedding B";
        variantMap["TSNE Settings B"] = _tsneSettingsActionB->toVariantMap();

        const auto probabilityDistributionB = _tsneAnalysisB.getProbabilityDistribution();

        if (_tsneSettingsActionB->getGeneralTsneSettingsAction().getSaveProbDistAction().isChecked() && probabilityDistributionB != std::nullopt)
        {
            const auto fileName = QUuid::createUuid().toString(QUuid::WithoutBraces) + ".bin";
            const auto filePath = QDir::cleanPath(projects().getTemporaryDirPath(AbstractProjectManager::TemporaryDirType::Save) + QDir::separator() + fileName).toStdString();

            std::ofstream saveFile(filePath, std::ios::out | std::ios::binary);

            if (!saveFile.is_open())
                std::cerr << "tsne embedding B Caching failed. File could not be opened. " << std::endl;
            else
            {
                hdi::data::IO::saveSparseMatrix(*probabilityDistributionB.value(), saveFile, nullptr);
                saveFile.close();
                variantMap["probabilityDistribution B"] = fileName;
            }
        }
    }

	if (_hsneSettingsAction != nullptr)
	{
        qDebug() << "toVariantMap() hsne was chosen for embedding B";
        variantMap["HSNE Settings B"] = _hsneSettingsAction->toVariantMap();

        if (_hsneSettingsAction->getHierarchyConstructionSettingsAction().getSaveHierarchyToProjectAction().isChecked() && _hierarchy->isInitialized())
        {
            // Handle HSNE Hierarchy
            {
                const auto fileName = QUuid::createUuid().toString(QUuid::WithoutBraces) + ".bin";
                const auto filePath = QDir::cleanPath(projects().getTemporaryDirPath(AbstractProjectManager::TemporaryDirType::Save) + QDir::separator() + fileName).toStdString();

                std::ofstream saveFile(filePath, std::ios::out | std::ios::binary);

                if (!saveFile.is_open())
                    std::cerr << "hsne embedding B Caching failed. File could not be opened. " << std::endl;
                else
                {
                    hdi::dr::IO::saveHSNE(_hierarchy->getHsne(), saveFile, nullptr);
                    saveFile.close();
                    variantMap["HsneHierarchy B"] = fileName;
                }
            }

            // Handle HSNE InfluenceHierarchy
            {
                const auto fileName = QUuid::createUuid().toString(QUuid::WithoutBraces) + ".bin";
                const auto filePath = QDir::cleanPath(projects().getTemporaryDirPath(AbstractProjectManager::TemporaryDirType::Save) + QDir::separator() + fileName).toStdString();

                _hierarchy->saveCacheHsneInfluenceHierarchy(filePath, _hierarchy->getInfluenceHierarchy().getMap());
                variantMap["HsneInfluenceHierarchy B"] = fileName;
            }
        }

        variantMap["selectionHelperDataGUID B"] = QVariant::fromValue(_selectionHelperData->getId());
	}


    return variantMap;
}

// =============================================================================
// Plugin Factory 
// =============================================================================

AnalysisPlugin* DualAnalysisPluginFactory::produce()
{
    return new DualAnalysisPlugin(this);
}

DualAnalysisPluginFactory::DualAnalysisPluginFactory()
{
    setIcon(StyledIcon(createPluginIcon("DUAL")));
}

PluginTriggerActions DualAnalysisPluginFactory::getPluginTriggerActions(const mv::Datasets& datasets) const
{
    PluginTriggerActions pluginTriggerActions;

    const auto getPluginInstance = [this](const Dataset<Points>& dataset) -> DualAnalysisPlugin* {
        return dynamic_cast<DualAnalysisPlugin*>(plugins().requestPlugin(getKind(), { dataset }));
    };

    if (PluginFactory::areAllDatasetsOfTheSameType(datasets, PointType)) {
        if (datasets.count() >= 1) {
            auto pluginTriggerAction = new PluginTriggerAction(const_cast<DualAnalysisPluginFactory*>(this), this, "Dual", "Perform dual analysis on selected datasets", icon(), [this, getPluginInstance, datasets](PluginTriggerAction& pluginTriggerAction) -> void {
                for (const auto& dataset : datasets)
                    getPluginInstance(dataset);
            });

            pluginTriggerActions << pluginTriggerAction;
        }

        if (datasets.count() >= 2) {
            auto pluginTriggerAction = new PluginTriggerAction(const_cast<DualAnalysisPluginFactory*>(this), this, "Group/Dual", "Group datasets and perform dual analysis on it", icon(), [this, getPluginInstance, datasets](PluginTriggerAction& pluginTriggerAction) -> void {
                getPluginInstance(mv::data().groupDatasets(datasets));
            });

            pluginTriggerActions << pluginTriggerAction;
        }
    }

    return pluginTriggerActions;
}

