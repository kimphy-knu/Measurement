/**
 * Local stub — replaces Firebase SDK for offline use.
 * Auth is always "logged in" as a local user.
 * Persistence uses localStorage instead of Firestore.
 */

export enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

export interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: object;
}

export function handleFirestoreError(error: unknown, operationType: OperationType, path: string | null) {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    operationType,
    path,
    authInfo: {},
  };
  console.error('Storage Error: ', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}

// Stub objects — not used directly; App.tsx uses localStorage instead
export const auth = { currentUser: null };
export const db = {};
