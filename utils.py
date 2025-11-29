from matplotlib import pyplot as plt


def plot_results(results):
    """Visualizes the results of an experiment."""
    train_losses, test_losses, train_accs, test_accs, is_shuffled = results
    title_suffix = "(Shuffle=True)" if is_shuffled else "(Shuffle=False)"

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))

    # Plot Losses
    ax[0].plot(train_losses, label='Train')
    ax[0].plot(test_losses, label='Test')
    ax[0].set_title(f'Losses {title_suffix}')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # Plot Accuracy
    ax[1].plot(train_accs, label='Train')
    ax[1].plot(test_accs, label='Test')
    ax[1].set_title(f'Accuracy {title_suffix}')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].legend()

    plt.show()
