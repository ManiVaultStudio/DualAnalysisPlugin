#include "SettingsAction.h"

#include <QHBoxLayout>

using namespace mv::gui;

SettingsAction::SettingsAction(QObject* parent, const QString& title) :
    GroupAction(parent, title),
    _embeddingAlgorithmAction(this, "Embedding algorithm"),
    _alignmentAction(this, "Align")
{
    _embeddingAlgorithmAction.setToolTip("Choose embedding algorithm");

    _embeddingAlgorithmAction.initialize(QStringList({ "None", "tSNE", "HSNE" }), "None");
    addAction(&_embeddingAlgorithmAction);

    _alignmentAction.setToolTip("Align the four embeddings");
    addAction(&_alignmentAction);

}

void SettingsAction::fromVariantMap(const QVariantMap& variantMap)
{
    WidgetAction::fromVariantMap(variantMap);

    _embeddingAlgorithmAction.fromParentVariantMap(variantMap);
    
    _alignmentAction.fromParentVariantMap(variantMap);  
    
}

QVariantMap SettingsAction::toVariantMap() const
{
    QVariantMap variantMap = WidgetAction::toVariantMap();

    _embeddingAlgorithmAction.insertIntoVariantMap(variantMap);

    _alignmentAction.insertIntoVariantMap(variantMap);
   
    return variantMap;
}

