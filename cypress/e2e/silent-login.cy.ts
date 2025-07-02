// eslint-disable-next-line @typescript-eslint/triple-slash-reference
/// <reference path="../support/index.d.ts" />
import { adminUser } from '../support/e2e';

describe('OAuth silent login', () => {
    before(() => {
        cy.registerAdmin();
    });

    it('redirects to oauth login when silent login enabled', () => {
        cy.loginAdmin();
        cy.request('/api/v1/configs/export').then((res) => {
            const cfg = res.body;
            cfg.oauth = cfg.oauth || {};
            cfg.oauth.silent_login = true;
            cy.request('POST', '/api/v1/configs/import', { config: cfg });
        });
        cy.intercept('GET', '/oauth/oidc/login*').as('silent');
        cy.visit('/auth');
        cy.wait('@silent').its('request.url').should('include', 'silent=true');
    });
});
