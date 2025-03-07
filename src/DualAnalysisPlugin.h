#pragma once

#include <AnalysisPlugin.h>
#include <event/EventListener.h>

#include <Task.h>

#include "PointData/PointData.h"

#include "TsneAnalysis.h"

#include "HSNE/HsneHierarchy.h"
#include "HSNE/HsneSettingsAction.h"

#include "Actions/SettingsAction.h"

// embedding alignment
#include "hdi/dimensionality_reduction/embedding_equalizer.h"

using namespace mv::plugin;
using namespace mv::gui;

class Points;

class TsneSettingsAction;

class HsneScaleAction;

class DualAnalysisPlugin : public AnalysisPlugin
{
    Q_OBJECT
public:
    DualAnalysisPlugin(const PluginFactory* factory);
    ~DualAnalysisPlugin() override;

    void init() override;

    // tSNE - for without master, without initial embedding
    void startComputation(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction, mv::Task& dataPreparationTask);
    
    // Modified to take in a master embedding (2D) for computing 1D embedding, initial embedding is the y -axis of the input
    void startComputation(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction, mv::Task& dataPreparationTask,
        hdi::data::Embedding<float>& masterEmbedding);

    // Modify based on startComputation to iterate embedding based on master embedding and connections
    void startAlignmentComputation(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction, mv::Task& dataPreparationTask,
        		hdi::data::Embedding<float>& masterEmbedding);
    
    void reinitializeComputation(mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction, ProbDistMatrix& probDistMatrix);
    void continueComputation(mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction, ProbDistMatrix& probDistMatrix, mv::Task& dataPreparationTask);
    void stopComputation(TsneAnalysis& tsneAnalysis);

    // HSNE
    void computeTopLevelEmbedding(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis);
    void continueComputation(mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis);  // TODO: same function name as tsne, should be renamed

    HsneHierarchy& getHierarchy() { return *_hierarchy.get(); }
    TsneAnalysis& getTsneAnalysis() { return _tsneAnalysisHSNEB; }

    HsneSettingsAction& getHsneSettingsAction() { return *_hsneSettingsAction; }

    // Event handling
    void onRefineFinished(mv::Dataset<Points> refineEmbedding);

    void onAlignmentTriggered();


private:
    void setup2DTsneForDataset(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsneSettingsAction,
        ProbDistMatrix& probDistMatrix, mv::Task& dataPreparationTask);

    void transposeData();

    // data initialization
    void initializeEmbeddingA();

    void initializeEmbeddingB();


    // compute 1D Tsne without master embedding - not used for now
    void compute1DTsne(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsne1DSettingsAction, mv::Task& dataPreparationTask);
    // compute 1D Tsne using alignment with 2D embedding as master
    void compute1DTsne(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis, TsneSettingsAction*& tsne1DSettingsAction, mv::Task& dataPreparationTask,
        hdi::data::Embedding<float>& masterEmbedding);

    void setupHSNEForDataset(mv::Dataset<Points>& inputDataset, mv::Dataset<Points>& embeddingDataset, TsneAnalysis& tsneAnalysis);

    hdi::data::Embedding<float> convertDatasetToEmbedding(const mv::Dataset<Points>& dataset, int dimensionality);

    // after drill in , compute 2D/1D embedding of A based on refined data
    void compute2DEmbeddingAWhenDrillInB();

    void compute1DEmbeddingAWhenDrillInB();

    // test functions for align 4 embeddings
    void align2DAto1DA();


public: // Serialization

    /**
     * Load plugin from variant map
     * @param Variant map representation of the plugin
     */
    Q_INVOKABLE void fromVariantMap(const QVariantMap& variantMap) override;

    /**
     * Save plugin to variant map
     * @return Variant map representation of the plugin
     */
    Q_INVOKABLE QVariantMap toVariantMap() const override;

signals:
    // Local signals for HSNE
    void startHierarchyWorker();

private:
    mv::Dataset<Points>                 _datasetB;			   /** Dataset B - cell by gene matrix*/
    mv::Dataset<Points>                 _datasetA;			   /** Dataset A - gene by cell matrix - transposed by datasetB*/

    mv::Dataset<Points>                 _embedding2DDatasetA;  /** 2D embedding of dataset A */
    mv::Dataset<Points>                 _embedding2DDatasetB;  /** 2D embedding of dataset B */

    mv::Dataset<Points>                 _embedding1DDatasetA;  /** 1D embedding of dataset A */
    mv::Dataset<Points>                 _embedding1DDatasetB;  /** 1D embedding of dataset B */

    SettingsAction                      _settingsAction;

    // 2D tSNE B
    TsneAnalysis                        _tsneAnalysisB;          /** TSNE analysis - test use this for tsne and hsne*/ 
    TsneSettingsAction*                 _tsneSettingsActionB;    /** TSNE settings action */
    ProbDistMatrix                      _probDistMatrixB;        /** Probability distribution matrix used for serialization */
    mv::Task                            _dataPreparationTaskB;   /** Task for reporting data preparation progress */

    // 2D tSNE A
    TsneAnalysis                        _tsneAnalysisA;          /** TSNE analysis */
    TsneSettingsAction*                 _tsneSettingsActionA;    /** TSNE settings action */
    ProbDistMatrix                      _probDistMatrixA;        /** Probability distribution matrix used for serialization */
    mv::Task                            _dataPreparationTaskA;   /** Task for reporting data preparation progress */

    // 1D tSNE B
    TsneAnalysis                        _1DtsneAnalysisB;          /** TSNE analysis */
    TsneSettingsAction*                 _1DtsneSettingsActionB;    /** TSNE settings action */
    ProbDistMatrix                      _1DprobDistMatrixB;        /** Probability distribution matrix used for serialization */
    mv::Task                            _1DdataPreparationTaskB;   /** Task for reporting data preparation progress */

    // 1D tSNE A
    TsneAnalysis                        _1DtsneAnalysisA;          /** TSNE analysis */
    TsneSettingsAction*                 _1DtsneSettingsActionA;    /** TSNE settings action */
    ProbDistMatrix                      _1DprobDistMatrixA;        /** Probability distribution matrix used for serialization */
    mv::Task                            _1DdataPreparationTaskA;   /** Task for reporting data preparation progress */

    // HSNE B
    std::unique_ptr<HsneHierarchy>      _hierarchy;      /** HSNE hierarchy */
    QThread                             _hierarchyThread;       /** Qt Thread for managing HSNE hierarchy computation */
    TsneAnalysis                        _tsneAnalysisHSNEB;          /** TSNE analysis */
    HsneSettingsAction*                 _hsneSettingsAction;    /** Pointer to HSNE settings action */
    mv::Dataset<Points>                 _selectionHelperData;   /** Invisible selection helper dataset */


    // recomputed embedding A based on refined embedding B
    std::vector<mv::Dataset<Points>>      _recomputedEmbedding2DDatasetsA;  // recomputed 2D embedding A based on refined selection
    std::vector<mv::Dataset<Points>>      _recomputedEmbedding1DDatasetsA; 
    std::vector<mv::Dataset<Points>>      _refinedDatasetsA;  // refined datasets A

    std::vector<mv::Dataset<Points>>      _embedding1DRefinedDatasets;  // refined 1D datasets B: tsne embedding of hsne embedding B

    int                          _numRecomputedEmbeddings = 0;



    // align 1D to 2D 
    hdi::data::Embedding<float>		   _embedding2DA;  // A
    hdi::data::Embedding<float>		   _embedding2DB;  // B
    hdi::data::Embedding<float>        _refinedEmbedding2D;  //  refined



    // align A to B
    std::vector<std::unordered_map<uint32_t, uint32_t>> _connectionsAB;
    hdi::data::Embedding<float>		   _embedding1DA;  // A
    hdi::data::Embedding<float>		   _embedding1DB;  // B

    bool                                _triggerAlignment = 0;

    bool                               _loadingFromProject = false;
    
};

class DualAnalysisPluginFactory : public AnalysisPluginFactory
{
    Q_INTERFACES(mv::plugin::AnalysisPluginFactory mv::plugin::PluginFactory)
        Q_OBJECT
        Q_PLUGIN_METADATA(IID   "nl.tudelft.DualAnalysisPlugin"
                          FILE  "DualAnalysisPlugin.json")

public:

    DualAnalysisPluginFactory();

    ~DualAnalysisPluginFactory() override {};

    /**
     * Produces the plugin
     * @return Pointer to the produced plugin
     */
    AnalysisPlugin* produce() override;

    /**
     * Get plugin trigger actions given \p datasets
     * @param datasets Vector of input datasets
     * @return Vector of plugin trigger actions
     */
    PluginTriggerActions getPluginTriggerActions(const mv::Datasets& datasets) const override;
};
