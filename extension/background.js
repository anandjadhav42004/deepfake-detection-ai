chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'scan-with-antigravity',
    title: 'Scan with AntiGravity',
    contexts: ['selection', 'page']
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const text = info.selectionText || info.pageUrl || '';
  if (!text) {
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icon128.png',
      title: 'AntiGravity',
      message: 'No text or URL selected.'
    });
    return;
  }

  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icon128.png',
    title: 'AntiGravity',
    message: 'Context scan opened in popup.'
  });
});
