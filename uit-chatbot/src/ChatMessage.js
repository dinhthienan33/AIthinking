import React from 'react';
import { Box, Typography } from '@material-ui/core';

function ChatMessage({ message }) {
  const isBot = message.sender === 'bot';

  return (
    <Box
      display="flex"
      justifyContent={isBot ? 'flex-start' : 'flex-end'}
      mb={2}
    >
      <Box
        bgcolor={isBot ? '#e0e0e0' : '#2196f3'}
        color={isBot ? 'text.primary' : 'white'}
        p={2}
        borderRadius={16}
        maxWidth="70%"
      >
        <Typography>{message.text}</Typography>
      </Box>
    </Box>
  );
}

export default ChatMessage;