#pragma once

#include <actions/GroupAction.h>

#include <actions/OptionAction.h>

#include <actions/TriggerAction.h>


using namespace mv::gui;

/**
 * Settings action class
 *
 * Action class for configuring settings
 */
class SettingsAction : public GroupAction
{
public:
    /**
     * Construct with \p parent object and \p title
     * @param parent Pointer to parent object
     * @param title Title
     */
    Q_INVOKABLE SettingsAction(QObject* parent, const QString& title);

    ///**
    // * Get action context menu
    // * @return Pointer to menu
    // */
    //QMenu* getContextMenu();

public: // Serialization

    /**
     * Load widget action from variant map
     * @param Variant map representation of the widget action
     */
    void fromVariantMap(const QVariantMap& variantMap) override;

    /**
     * Save widget action to variant map
     * @return Variant map representation of the widget action
     */

    QVariantMap toVariantMap() const override;

public: // Action getters

    OptionAction& getEmbeddingAlgorithmAction() { return _embeddingAlgorithmAction; }

    TriggerAction& getAlignmentAction() { return _alignmentAction; }
   

private:
    OptionAction   _embeddingAlgorithmAction; /** Action to choose embedding algorithm */

    TriggerAction  _alignmentAction; /** Action to align the four embeddings */

};
