// Main frontend utilities for AI Content Factory

document.addEventListener('DOMContentLoaded', () => {
    // Copy to clipboard utility
    window.copyText = function(text, buttonElement) {
        if (!navigator.clipboard) {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            showCopiedFeedback(buttonElement);
            return;
        }

        navigator.clipboard.writeText(text).then(() => {
            showCopiedFeedback(buttonElement);
        }).catch(err => {
            console.error('Failed to copy: ', err);
        });
    };

    function showCopiedFeedback(buttonElement) {
        if (!buttonElement) return;
        const originalHtml = buttonElement.innerHTML;
        buttonElement.innerHTML = '<i class="bi bi-check2"></i> Đã chép!';
        buttonElement.classList.add('btn-success');
        buttonElement.classList.remove('btn-outline-secondary', 'btn-primary');
        setTimeout(() => {
            buttonElement.innerHTML = originalHtml;
            buttonElement.classList.remove('btn-success');
            buttonElement.classList.add('btn-outline-secondary');
        }, 1800);
    }
});
